from typing import Optional
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import ExponentialLR
from omegaconf import OmegaConf
import os
import json
import argparse
import lightning.pytorch as L
import math
import torch.nn.functional as F
import random


# reads from a JSONL file, example:
"""
{"word": "page", "kata": ["ページ"]}
{"word": "spring", "kata": ["スプリング"]}
{"word": "costume", "kata": ["コスチューム"]}
{"word": "article", "kata": ["アーテクル", "アーティクル"]}
{"word": "spain", "kata": ["イスパニア", "エスパニア", "スペイン"]}
"""


class E2KDataset(Dataset):
    def __init__(self, path: str, eval: bool = False, causal: bool = False):
        self.flat_data = []  # [word1: kata1, word1: kata2, word2: kata1, ...]
        self.data = {}  # {word1: [kata1, kata2], word2: [kata1], ...}
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        self.input_symbols = set()
        self.output_symbols = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                word = item["word"]
                kata_list = item["kata"]
                self.data[word] = kata_list
                for kata in kata_list:
                    self.flat_data.append((word, kata))
                    for ch in word:
                        self.input_symbols.add(ch)
                    for ch in kata:
                        self.output_symbols.add(ch)
        self.flat_data = list(self.flat_data)
        self.data = list(self.data.items())
        if causal:
            self.update_symbols(
                ["<pad>", "<sos>", "<eos>"] + sorted(self.input_symbols),
                ["<pad>", "<sos>", "<eos>"] + sorted(self.output_symbols),
            )
        else:
            self.update_symbols(
                ["<pad>"] + sorted(self.input_symbols),
                ["<pad>"] + sorted(self.output_symbols),
            )
        self.eval = eval
        self.causal = causal

    def update_symbols(self, new_isyms, new_osyms):
        self.input_symbols = new_isyms
        self.output_symbols = new_osyms
        self.input_symbol_to_idx = {s: i for i, s in enumerate(self.input_symbols)}
        self.output_symbol_to_idx = {s: i for i, s in enumerate(self.output_symbols)}

    def __len__(self):
        if self.eval:
            return len(self.data)
        else:
            return len(self.flat_data)

    def __getitem__(self, idx):
        if self.eval:
            # returns the multi data as plain strings
            return self.data[idx]
        else:
            # convert the strings to Tensors of indices
            word, kata = self.flat_data[idx]
            if self.causal:
                word = ["<sos>", *list(word), "<eos>"]
                kata = ["<sos>", *list(kata), "<eos>"]
            word_indices = torch.tensor(
                [self.input_symbol_to_idx[ch] for ch in word], dtype=torch.long
            )
            kata_indices = torch.tensor(
                [self.output_symbol_to_idx[ch] for ch in kata], dtype=torch.long
            )
            return word_indices, kata_indices


def lens2mask(lengths):
    max_len = lengths.max()
    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0)
    mask = ids >= lengths.unsqueeze(1)
    return mask  # (B, max_len) bool tensor, True for padded positions


def collate_fn(batch):
    src, tgt = zip(*batch, strict=False)
    src_lens = [s.shape[0] for s in src]
    tgt_lens = [t.shape[0] for t in tgt]
    src_padded = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=0)
    src_mask = lens2mask(torch.tensor(src_lens, device=src_padded.device))
    tgt_mask = lens2mask(torch.tensor(tgt_lens, device=tgt_padded.device))
    return src_padded, tgt_padded, src_mask, tgt_mask


class SinPE(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinPE, self).__init__()
        pe = self.positionalencoding1d(d_model, max_len)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def positionalencoding1d(self, d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(d_model)
            )
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp(
            (
                torch.arange(0, d_model, 2, dtype=torch.float)
                * -(math.log(10000.0) / d_model)
            )
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].clone().detach()
        return x


class MLME2KModel(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, cfg):
        super().__init__()
        m_cfg = cfg.model
        self.cfg = cfg
        self.CLS_IDX = src_vocab_size  # last index for [CLS] token so we don't need to shift the src indices
        self.MASK_IDX = tgt_vocab_size  # last index for [MASK] token as above
        self.src_emb = nn.Embedding(src_vocab_size + 1, m_cfg.dim)  # +1 for [CLS] token
        self.tgt_emb = nn.Embedding(tgt_vocab_size + 1, m_cfg.dim)
        self.pe = SinPE(m_cfg.dim, max_len=m_cfg.max_len)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=m_cfg.dim, nhead=m_cfg.n_heads, batch_first=True
            ),
            num_layers=m_cfg.n_enc_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=m_cfg.dim, nhead=m_cfg.n_heads, batch_first=True
            ),
            num_layers=m_cfg.n_dec_layers,
        )
        self.length_predictor = nn.Linear(m_cfg.dim, m_cfg.max_len)
        self.out = nn.Linear(m_cfg.dim, tgt_vocab_size)

    def generate_tgt_mask(self, tgt: Tensor, ratio: float):
        """
        Generate a target mask for the decoder with a given masking ratio.
        ratio: a float between 0 and 1 indicating the proportion of tokens to mask.
        for example, ratio=0.3 means 30% of the tokens will be masked (set True in the mask)
        """
        B, T = tgt.size()
        mask = torch.rand(B, T, device=tgt.device) < ratio  # (B, T)
        return mask  # bool tensor

    def forward(
        self, src: Tensor, tgt: Tensor, src_mask: Tensor = None, tgt_mask: Tensor = None
    ):
        """
        Masked LM task
        1. encoder encodes the source sequence with an appended [CLS] token
        2. get the [CLS] token output as the length of the target sequence, a CrossEntropyLoss is applied here
        3. the target sequence is masked by a random mask, and the decoder predicts the original target sequence
        4. CrossEntropyLoss is applied on ONLY the masked positions
        """
        src = F.pad(src, (1, 0), value=self.CLS_IDX)  # prepend [CLS] token
        # also add 1 to src_mask if provided
        if src_mask is not None:
            src_mask = F.pad(src_mask, (1, 0), value=False)
        src_emb = self.pe(self.src_emb(src))
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)  # (B, T_src+1, D)
        # predict length
        cls_output = memory[:, 0, :]  # (B, D)
        memory = memory[:, 1:, :]  # (B, T_src, D)
        if src_mask is not None:
            src_mask = src_mask[:, 1:]  # (B, T_src)
        length_logits = self.length_predictor(cls_output)  # (B, max_len)
        # prepare tgt mask
        rand_mask = self.generate_tgt_mask(
            tgt, ratio=self.cfg.train.mask_ratio
        )  # (B, T_tgt)
        # decoder
        masked_tgt = tgt.masked_fill(rand_mask, self.MASK_IDX)
        # fill back the padding positions
        if tgt_mask is not None:
            masked_tgt = masked_tgt.masked_fill(tgt_mask, 0)
        masked_tgt_emb = self.pe(self.tgt_emb(masked_tgt))
        decoder_output = self.decoder(
            masked_tgt_emb,
            memory,
            tgt_is_causal=False,
            memory_is_causal=False,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=src_mask,
        )
        output_logits = self.out(decoder_output)  # (B, T_tgt, tgt_vocab_size)
        return length_logits, output_logits, rand_mask

    def generate(
        self, src: Tensor, src_mask: Optional[Tensor] = None
    ) -> tuple[Tensor, int]:
        """
        1. encode the source sequence with an appended [CLS] token
        2. get the length from the [CLS] token output
        3. generate a all-[MASK] target sequence with the predicted length
        4. decode to get the output sequence
        5. find the most confident token (with highest probability)
        6. unmask and assign the token to the output sequence
        7. repeat 4-6 until all tokens are unmasked

        For simplicity, we assume the src and tgt are unbatched (B=1), thus no padding masks are needed.
        """
        src = F.pad(src, (1, 0), value=self.CLS_IDX)  # prepend [CLS] token
        src_emb = self.pe(self.src_emb(src))
        if src_mask is not None:
            src_mask = F.pad(src_mask, (1, 0), value=False)
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        # predict length
        cls_output = memory[:, 0, :]  # (1, D)
        memory = memory[:, 1:, :]  # (1, T_src, D)
        if src_mask is not None:
            src_mask = src_mask[:, 1:]  # (1, T_src)
        length_logits = self.length_predictor(cls_output)  # (1, max_len)
        pred_length = torch.argmax(length_logits, dim=1).item()  # scalar
        # prepare all-MASK target
        generated = torch.full(
            (1, pred_length), self.MASK_IDX, dtype=torch.long, device=src.device
        )  # (1, pred_length)
        unmasked = torch.zeros(
            1, pred_length, dtype=torch.bool, device=src.device
        )  # (1, pred_length)
        for _ in range(pred_length):
            tgt_emb = self.pe(self.tgt_emb(generated))
            decoder_output = self.decoder(
                tgt_emb,
                memory,
                tgt_is_causal=False,
                memory_is_causal=False,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=src_mask,
            )
            output_logits = self.out(decoder_output)  # (1, pred_length, tgt_vocab_size)
            probs = F.softmax(output_logits, dim=-1)  # (1, pred_length, tgt_vocab_size)
            # set already unmasked positions to 0 prob
            probs = probs.masked_fill(
                unmasked.unsqueeze(-1), 0.0
            )  # (1, pred_length, tgt_vocab_size)
            # find the most confident token
            max_probs, max_indices = torch.max(probs, dim=-1)  # (1, pred_length)
            # find the position with the highest confidence
            pos = torch.argmax(max_probs, dim=-1).item()  # scalar
            token = max_indices[0, pos].item()
            # unmask and assign the token
            generated[0, pos] = token
            unmasked[0, pos] = True
        return generated.squeeze(0), pred_length


class CausalE2KModel(nn.Module):
    """
    A traditional seq2seq model with causal decoding for comparison.
    """

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, cfg):
        super().__init__()
        m_cfg = cfg.model
        self.cfg = cfg
        self.src_emb = nn.Embedding(src_vocab_size, m_cfg.dim)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, m_cfg.dim)
        self.pe = SinPE(m_cfg.dim, max_len=m_cfg.max_len)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=m_cfg.dim, nhead=m_cfg.n_heads, batch_first=True
            ),
            num_layers=m_cfg.n_enc_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=m_cfg.dim, nhead=m_cfg.n_heads, batch_first=True
            ),
            num_layers=m_cfg.n_dec_layers,
        )
        self.out = nn.Linear(m_cfg.dim, tgt_vocab_size)
        self.sos_idx = 1
        self.eos_idx = 2

    def forward(
        self, src: Tensor, tgt: Tensor, src_mask: Tensor = None, tgt_mask: Tensor = None
    ):
        tgt = tgt[:, :-1]  # remove the last token for teacher forcing
        # fill back the padding positions
        if tgt_mask is not None:
            tgt_mask = tgt_mask[:, :-1]

        src_emb = self.pe(self.src_emb(src))
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)  # (B, T_src, D)
        tgt_emb = self.pe(self.tgt_emb(tgt))

        # Generate causal mask for the target sequence
        tgt_seq_len = tgt.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_seq_len, device=tgt.device
        )
        decoder_output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=causal_mask,  # Pass the causal mask here
            tgt_is_causal=True,
            memory_is_causal=False,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=src_mask,
        )
        output_logits = self.out(decoder_output)  # (B, T_tgt, tgt_vocab_size)
        return output_logits

    def generate(self, src: Tensor, src_mask: Optional[Tensor] = None):
        max_len = self.cfg.model.max_len
        src_emb = self.pe(self.src_emb(src))
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        generated = [self.sos_idx]
        for _ in range(max_len):
            tgt_input = torch.tensor(
                generated, dtype=torch.long, device=src.device
            ).unsqueeze(0)  # (1, i+1)
            tgt_emb = self.pe(self.tgt_emb(tgt_input))
            decoder_output = self.decoder(
                tgt_emb,
                memory,
                tgt_is_causal=True,
                memory_is_causal=False,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=src_mask,
            )
            output_logits = self.out(decoder_output)  # (1, i+1, tgt_vocab_size+2)
            next_token_logits = output_logits[0, -1, :]  # (tgt_vocab_size+2)
            next_token = torch.argmax(next_token_logits).item()
            if next_token == self.eos_idx:
                break
            generated.append(next_token)
        return torch.tensor(generated, dtype=torch.long, device=src.device)[1:], len(
            generated
        )


class MLME2KLightningModule(L.LightningModule):
    def __init__(self, cfg, in_symbols, out_symbols):
        super().__init__()
        self.model = MLME2KModel(
            src_vocab_size=len(in_symbols),
            tgt_vocab_size=len(out_symbols),
            cfg=cfg,
        )
        self.cfg = cfg
        self.in_symbols = in_symbols
        self.out_symbols = out_symbols

    def training_step(self, batch, batch_idx):
        src, tgt, src_mask, tgt_mask = batch  # src: (B, T_src), tgt: (B, T_tgt)
        length_logits, output_logits, rand_mask = self.model.forward(
            src, tgt, src_mask, tgt_mask
        )
        # compute length loss
        tgt_lengths = (~tgt_mask).sum(dim=1)  # (B,)
        length_loss = F.cross_entropy(length_logits, tgt_lengths)
        # compute masked LM loss
        mlm_loss = (
            F.cross_entropy(
                output_logits.transpose(1, 2),  # (B, vocab_size, T_tgt)
                tgt,
                ignore_index=0,
                reduction="none",
            )
            .masked_fill(~rand_mask, 0.0)
            .sum()
            / (rand_mask & ~tgt_mask).sum()
        )
        total_loss = length_loss + mlm_loss
        self.log("train/loss", total_loss)
        self.log("train/length_loss", length_loss)
        self.log("train/mlm_loss", mlm_loss)
        return {
            "loss": total_loss,
            "length_loss": length_loss,
            "mlm_loss": mlm_loss,
        }

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            # print a sample generation
            src, tgt, src_mask, tgt_mask = batch
            random.seed(random.randint(0, int(1e3)))
            _batch_idx = random.randint(0, src.size(0) - 1)
            generated, pred_length = self.model.generate(
                src[_batch_idx].unsqueeze(0), src_mask[_batch_idx].unsqueeze(0)
            )
            print("Sample generation:")
            print(
                "Source:",
                "".join([self.in_symbols[idx] for idx in src[_batch_idx].tolist()]),
            )
            print(
                "Target:",
                "".join([self.out_symbols[idx] for idx in tgt[_batch_idx].tolist()]),
            )
            print(
                "Generated:",
                "".join([self.out_symbols[idx] for idx in generated.tolist()]),
            )
            print("Predicted length:", pred_length)
        loss_dict = self.training_step(batch, batch_idx)
        self.log("val/loss", loss_dict["loss"])
        self.log("val/length_loss", loss_dict["length_loss"])
        self.log("val/mlm_loss", loss_dict["mlm_loss"])
        return loss_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr)
        scheduler = ExponentialLR(optimizer, gamma=self.cfg.train.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class CausalE2KLightningModule(L.LightningModule):
    def __init__(self, cfg, in_symbols, out_symbols):
        super().__init__()
        self.model = CausalE2KModel(
            src_vocab_size=len(in_symbols),
            tgt_vocab_size=len(out_symbols),
            cfg=cfg,
        )
        self.cfg = cfg
        self.in_symbols = in_symbols
        self.out_symbols = out_symbols

    def training_step(self, batch, batch_idx):
        src, tgt, src_mask, tgt_mask = batch  # src: (B, T_src), tgt: (B, T_tgt)
        output_logits = self.model.forward(src, tgt, src_mask, tgt_mask)
        # compute seq2seq loss
        loss = F.cross_entropy(
            output_logits.transpose(1, 2),  # (B, vocab_size, T_tgt-1)
            tgt[:, 1:],  # remove the first token
            ignore_index=0,
        )
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        if batch_idx == 0:
            _batch_idx = random.randint(0, batch[0].size(0) - 1)
            src, tgt, src_mask, tgt_mask = batch
            generated, pred_length = self.model.generate(
                src[_batch_idx].unsqueeze(0), src_mask[_batch_idx].unsqueeze(0)
            )
            print("Sample generation (Causal Model):")
            print(
                "Source:",
                "".join([self.in_symbols[idx] for idx in src[_batch_idx].tolist()]),
            )
            print(tgt[_batch_idx].tolist())
            print(
                "Target:",
                "".join([self.out_symbols[idx] for idx in tgt[_batch_idx].tolist()]),
            )
            print(generated.tolist())
            print(
                "Generated:",
                "".join([self.out_symbols[idx] for idx in generated.tolist()]),
            )
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr)
        scheduler = ExponentialLR(optimizer, gamma=self.cfg.train.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def create_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    torch.manual_seed(cfg.seed)
    create_if_not_exists(cfg.output_dir)
    create_if_not_exists(cfg.tmp_dir)
    # prepare datasets
    splits = ["train_dataset.jsonl", "val_dataset.jsonl", "test_dataset.jsonl"]
    if not all([os.path.exists(os.path.join(cfg.tmp_dir, f)) for f in splits]):
        print("Splits not found, preparing datasets...")
        # save a split of the dataset for validation
        dataset = E2KDataset(os.path.join(cfg.tmp_dir, cfg.dataset.name), eval=True)
        train, val, test = random_split(
            dataset,
            [0.9, 0.05, 0.05],
            generator=torch.Generator().manual_seed(cfg.dataset.seed),
        )
        print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")
        for split, name in zip([train, val, test], splits):
            with open(os.path.join(cfg.tmp_dir, f"{name}"), "w", encoding="utf-8") as f:
                dumped = []
                for src, tgt in split:
                    item = {"word": src, "kata": tgt}
                    dumped.append(json.dumps(item, ensure_ascii=False))
                f.write("\n".join(dumped))
        print("Datasets prepared and saved to", cfg.tmp_dir)
    else:
        print("Dataset splits found, skipping preparation.")
    # load datasets
    train = E2KDataset(
        os.path.join(cfg.tmp_dir, "train_dataset.jsonl"), causal=cfg.model.causal
    )
    val = E2KDataset(
        os.path.join(cfg.tmp_dir, "val_dataset.jsonl"), causal=cfg.model.causal
    )
    test = E2KDataset(
        os.path.join(cfg.tmp_dir, "test_dataset.jsonl"), causal=cfg.model.causal
    )
    # val and test could be too small to cover all symbols, so we update their symbol tables from train set
    val.update_symbols(train.input_symbols, train.output_symbols)
    test.update_symbols(train.input_symbols, train.output_symbols)
    # training loop starts here
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        if cfg.model.causal:
            lit_model = CausalE2KLightningModule.load_from_checkpoint(
                args.resume,
                in_symbols=train.input_symbols,
                out_symbols=train.output_symbols,
                cfg=cfg,
            )
        else:
            lit_model = MLME2KLightningModule.load_from_checkpoint(
                args.resume,
                in_symbols=train.input_symbols,
                out_symbols=train.output_symbols,
                cfg=cfg,
            )
    else:
        if cfg.model.causal:
            lit_model = CausalE2KLightningModule(
                in_symbols=train.input_symbols,
                out_symbols=train.output_symbols,
                cfg=cfg,
            )
        else:
            lit_model = MLME2KLightningModule(
                in_symbols=train.input_symbols,
                out_symbols=train.output_symbols,
                cfg=cfg,
            )
    trainer = L.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="cpu",
        default_root_dir=cfg.output_dir,
        log_every_n_steps=10,
    )
    train_loader = DataLoader(
        train,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.dataset.num_workers,
    )
    val_loader = DataLoader(
        val,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.dataset.num_workers,
    )
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
