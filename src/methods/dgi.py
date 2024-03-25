import torch

from .base import BaseMethod
from src.augment import ShuffleNode, SumEmb
from src.losses import NegativeMI
from typing import Callable


class DGI(BaseMethod):
    r""" Deep Graph Infomax (DGI).

    Args:
        encoder (Optional[torch.nn.Module]): the encoder to be trained.
        hidden_channels (int): output dimension of the encoder.
        readout (str): "avg" or "max": generate global embeddings.
            (default: "avg")
    """
    def __init__(self,
                 encoder: torch.nn.Module,
                 hidden_channels: int,
                 readout: str="avg",
                 readout_act: Callable=torch.nn.Sigmoid()) -> None:
        super().__init__(encoder=encoder)

        self.readout = SumEmb(readout, readout_act)
        self.corrupt = ShuffleNode()
        self.loss_func = NegativeMI(in_channels=hidden_channels)

    def apply_data_augment(self, batch):
        batch = batch.to(self.device)
        batch2 = self.corrupt(batch).to(self.device)
        return batch, batch2

    def apply_emb_augment(self, h_pos):
        s = self.readout(h_pos, keepdim=True)
        return s

    def get_loss(self, h_pos, h_neg, s):
        s = s.expand_as(h_pos)
        loss = self.loss_func(x=s, y=h_pos, x_ind=s, y_ind=h_neg)
        return loss

    def forward(self, batch):
        # 1. data augmentation
        batch, batch2 = self.apply_data_augment(batch)

        # 2. get embeddings
        h_pos = self.encoder(batch)
        h_neg = self.encoder(batch2)

        # 3. emb augmentation
        s = self.apply_emb_augment(h_pos)

        # 4. get loss
        loss = self.get_loss(h_pos, h_neg, s)
        return loss
