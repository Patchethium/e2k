import lightning.pytorch as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from data import get_loaders
from model import Seq2Seq, ModelConfig
from e2k.constants import src_tokens, tgt_tokens
from random import randint

from einops import rearrange
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from omegaconf import OmegaConf


@dataclass
class TrainConfig:
    batch_size: int
    lr: float
    max_epochs: int
    decay: float
    log_interval: int
    num_workers: int


@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig


class E2KExperiement(L.LightningModule):
    def __init__(self, cfg: Config, test_loader: Optional[DataLoader] = None):
        super().__init__()
        self.cfg = cfg
        self.model = Seq2Seq(cfg.model)
        self.test_loader = test_loader

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor):
        return self.model(src, tgt, src_mask, tgt_mask)

    def inference(self, src: Tensor, src_mask: Tensor, cot: bool, max_len: int) -> Tensor:
        return self.model.inference(src, src_mask, cot, max_len)

    def _loss(self, logits: Tensor, tgt: Tensor) -> Tensor:
        return F.cross_entropy(logits[:, : -1].transpose(1, 2), tgt[:, 1:], ignore_index=0)

    def training_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ):
        src, tgt, src_mask, tgt_mask = batch
        logits = self(src, tgt, src_mask, tgt_mask)
        loss = self._loss(logits, tgt)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ):
        src, tgt, src_mask, tgt_mask = batch
        logits = self(src, tgt, src_mask, tgt_mask)
        loss = self._loss(logits, tgt)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int):
        with torch.inference_mode():
            self.eval()
            src, tgt, src_mask, tgt_mask = batch
            src, src_mask = [x.to(self.device) for x in [src, src_mask]]
            non_cot = self.inference(src, src_mask, False, 32)
            non_cot = torch.argmax(non_cot, dim=-1)[0] # [Tt]
            non_cot = [tgt_tokens[t] for t in non_cot]
            cot = self.inference(src, src_mask, True, 32)
            cot = torch.argmax(cot, dim=-1)[0]
            cot = [tgt_tokens[t] for t in cot]
            src = src[0] # [Ts]
            src = [src_tokens[s] for s in src]
            src = "".join(src)
            non_cot, cot =  [" ".join(x) for x in [non_cot, cot]]
            print(f"\n\nsrc: {src}\nnon_cot: {non_cot}\ncot: {cot}")
            self.train()

    def on_validation_epoch_end(self) -> None:
        """
        F*ck lightning, my homies do it ourselves
        """
        if self.test_loader is not None:
            sample = randint(0, len(self.test_loader) - 1)
            _iter = iter(self.test_loader)
            for _ in range(sample):
                _ = next(_iter)
            self.test_step(next(_iter), 0)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.cfg.train.lr)
        scheduler = ExponentialLR(optimizer, self.cfg.train.decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }

def main():
    cfg_path = "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg = Config(**cfg)
    train_loader, val_loader, test_loader = get_loaders(cfg.train.num_workers)
    model = E2KExperiement(cfg, test_loader)
    trainer = L.Trainer(max_epochs=cfg.train.max_epochs, log_every_n_steps=cfg.train.log_interval, accelerator="mps")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()
