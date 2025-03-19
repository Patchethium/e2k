import torch

from torch import nn, Tensor
import torch.nn.functional as F
from dataclasses import dataclass
from e2k.constants import src_tokens, tgt_tokens
from einops import rearrange


@dataclass
class ModelConfig:
    dim: int
    n_heads: int
    dropout: float


class Attention(nn.Module):
    def __init__(self, dim: int, n_head: int, dropoout: float=0.1):
        self.n_head = n_head
        self.scale = dim ** -0.5
        self.d_head = dim // n_head
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropoout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, beta: Tensor, src_mask: Tensor, tgt_mask: Tensor):
        """
        q: [B,Tq,N]
        k: [B,Tk,N]
        v: [B,Tk,N]
        beta: [H,Tq,Tk]
        src_mask: [B,Tk], True for valid
        tgt_mask: [B,Tq], same
        """
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        q = rearrange(q, "B T (H D) -> B H T D", H = self.n_head, D=self.d_head)
        k = rearrange(k, "B T (H D) -> B T H D", H = self.n_head, D=self.d_head)
        v = rearrange(v, "B T (H D) -> B H T D", H = self.n_head, D=self.d_head)
        attn = q @ k * self.scale # [B,Tq,Tk]
        attn_mask = mask.unsqueeze(2) & src_mask.unsqueeze(1) # [B,Tq,Tk]
        attn_mask = mask.unsqueeze(1) # [B,1,Tq,Tk]
        attn.masked_fill_(attn_mask, 1e-9) # [B,H,Tq,Tk]
        attn = F.softmax(attn, dim=-1)
        attn += beta.unsqueeze(0)
        attn = dropout(attn)
        v = attn @ v # [B,H,Tq,N]
        v = rearrange(v, "B H T N -> B T (H N)", H=self.n_head)
        return v

class BaseTransformerLayer(nn.Module):
    def __init__(self, dim: int, n_head: int, dropout: float):
        super().__init__()
        self.attn = Attn(dim, n_head, dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2)])
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(4*dim, dim)
        )

    def forward(self):
        raise UnImplementedExpection

class TransformerEncoder(BaseTransformerLayer):
    def forward(self, x: Tensor, beta: Tensor, mask:Tensor):
        x = self.attn(x,x,x,beta,mask,mask) + x
        x = self.norms[0](x)
        x = self.ffn(x) + x
        x = self.norms[1](x)
        return x

class TransformerDecoder(BaseTransformerLayer):
    def __init__(self, dim: int, n_head: int, dropout: float):
        super().__init__(self, dim, n_head, dropout)
        self.cross_attn = Attn(dim, n_head, dropout)
        self.norms.append(nn.LayerNorm(dim))

    def forward(self, src:Tensor, tgt:Tensor, beta: Tensor, src_mask:Tensor, tgt_mask: Tensor):
        x = self.attn(tgt, tgt, tgt, beta, tgt_mask, tgt_mask) + x
        x = self.norms[0](x)
        x = self.cross_attn(tgt, src, src, beta, src_mask, tgt_mask) + x
        x = self.norms[1](x)
        x = self.ffn(x) + x
        x = self.norms[2](x)
        return x

class Seq2Seq(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        src_dim = len(src_tokens)
        tgt_dim = len(tgt_tokens)
        self.src_idx = {c: i for i, c in enumerate(src_tokens)}
        self.tgt_idx = {c: i for i, c in enumerate(tgt_tokens)}
        self.src_emb = nn.Embedding(src_dim, cfg.dim)
        self.tgt_emb = nn.Embedding(tgt_dim, cfg.dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.encoder = nn.ModuleList(
            [TransformerEncoder(cfg.dim, cfg.n_head, cfg.dropout) for _ in range(cfg.enc_layers)]
        )
        self.decoder = nn.ModuleList(
            [TransformerDecoder(cfg.dim, cfg.n_head, cfg.dropout) for _ in range(cfg.dec_layers)]
        )
        self.betas = nn.Parameter([2])

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor):
        """
        src: source sequence, [B,Ts]
        tgt: target sequence, [B,Tt]
        src_mask: source mask, [B,Ts], True for valid
        tgt_mask: unused, placed here for future use, same as above
        """
        src = self.src_emb(src)
        src = self.dropout(src)
        for layer in self.encoder:
            src = layer(src, src_mask)
        return logits

    def inference(self, src: Tensor, src_mask: Tensor, cot: bool, max_len: int) -> Tensor:
        """
        The source only inference, used for evaluation and testing
        src: source sequence, [B,Ts]
        src_mask: source mask, [B,Ts], True for valid
        cot: whether to use cot, if True, the first token will be <cot>, otherwise <sos>
        max_len: maximum length of the target sequence
        """
        B, _ = src.shape
        # encoder part is the same
        src_emb = self.src_emb(src)
        # src_emb = self.dropout(src_emb) # no dropout in inference
        enc_o, enc_h = self.encoder(src_emb)
        enc_o = self.enc_post(enc_o)
        enc_h = torch.cat([enc_h[0], enc_h[1]], dim=-1).unsqueeze(0)
        enc_h = self.enc_h_post(enc_h)

        # decoder part
        logits = torch.zeros(B, max_len, len(tgt_tokens), device=src.device)
        last_token = "<cot>" if cot else "<sos>"
        last_token = torch.tensor([self.tgt_idx[last_token]], device=src.device).unsqueeze(0).expand(B, 1) # [B,1]
        for i in range(max_len):
            tgt_emb = self.tgt_emb(last_token)
            # tgt_emb = self.dropout(tgt_emb) # no dropout in inference
            dec_o, enc_h = self.decoder(tgt_emb, enc_h) # don't forget to update the hidden state
            attn, _ = self.attn.forward(tgt_emb, enc_o, enc_o, key_padding_mask=~src_mask)
            post_in = torch.cat([dec_o, attn], dim=-1)
            logits_i = self.post(post_in) # [B,1,V]
            last_token = torch.argmax(logits_i, dim=-1) # [B,1]
            logits[:, i] = logits_i.squeeze(1) # [B,V]
        return logits
