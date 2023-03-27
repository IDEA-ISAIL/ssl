import torch

from .base import BaseMethod, ContrastiveMethod
from .utils import AvgReadout
from src.augment import ShuffleNode, Echo
from src.losses import NegativeMI

from typing import Optional, Callable, Union
from src.typing import AugmentType


class DGI(BaseMethod):
    r""" Deep Graph Infomax (DGI).

    Args:
        encoder (Optional[torch.nn.Module]): the encoder to be trained.
        hidden_channels (int): output dimension of the encoder.
        readout (Union[Callable, torch.nn.Module]): the readout function to obtain the summary emb. of the entire graph.
            (default: AvgReadout())
        corruption (OptAugment): data augmentation/corruption to generate negative node pairs.
            (default: ShuffleNode())
        loss_function (Optional[torch.nn.Module]): the loss function. If None, then use the NegativeMI loss.
    """
    def __init__(self,
                 encoder: torch.nn.Module,
                 hidden_channels: int,
                 readout: Union[Callable, torch.nn.Module] = AvgReadout(),
                 corruption: AugmentType = ShuffleNode(),
                 loss_function: Optional[torch.nn.Module] = None) -> None:
        loss_function = loss_function if loss_function else NegativeMI(hidden_channels)
        super().__init__(encoder=encoder, data_augment=corruption, loss_function=loss_function)

        self.readout = readout
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, batch1, batch2):
        edge_weight = batch1.edge_weight if "edge_weight" in batch1 else None
        h_pos = self.encoder(x=batch1.x, edge_index=batch1.edge_index, edge_weight=edge_weight)
        h_neg = self.encoder(x=batch2.x, edge_index=batch2.edge_index, edge_weight=edge_weight)

        s = self.readout(h_pos, keepdim=True)
        s = self.sigmoid(s)

        s = s.expand_as(h_pos)
        pos_pairs = torch.stack([s, h_pos], -1)
        neg_pairs = torch.stack([s, h_neg], -1)
        return pos_pairs, neg_pairs

    def train_iter(self, batch):
        """TODO: maybe we can do some further generalization for contrastive methods here."""
        batch2 = self.data_augment(batch).to(self._device)
        pos_pairs, neg_pairs = self.forward(batch1=batch, batch2=batch2)
        loss = self.loss_function(pos_pairs=pos_pairs, neg_pairs=neg_pairs)
        return loss


class DGI2(ContrastiveMethod):
    r""" Deep Graph Infomax (DGI).

    Args:
        encoder (Optional[torch.nn.Module]): the encoder to be trained.
        hidden_channels (int): output dimension of the encoder.
        readout (Union[Callable, torch.nn.Module]): the readout function to obtain the summary emb. of the entire graph.
            (default: AvgReadout())
        corruption (OptAugment): data augmentation/corruption to generate negative node pairs.
            (default: ShuffleNode())
        loss_function (Optional[torch.nn.Module]): the loss function. If None, then use the NegativeMI loss.
    """
    def __init__(self,
                 encoder: torch.nn.Module,
                 hidden_channels: int,
                 readout: Union[Callable, torch.nn.Module] = AvgReadout(),
                 corruption: AugmentType = ShuffleNode(),
                 loss_function: Optional[torch.nn.Module] = None) -> None:
        loss_function = loss_function if loss_function else NegativeMI(hidden_channels)
        super().__init__(encoder=encoder, data_augment={"1": Echo(), "2": corruption}, loss_function=loss_function)

        self.readout = readout
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, aug2data):
        batch1, batch2 = aug2data["1"], aug2data["2"]
        edge_weight = batch1.edge_weight if "edge_weight" in batch1 else None
        h1 = self.encoder(x=batch1.x, edge_index=batch1.edge_index, edge_weight=edge_weight)
        h2 = self.encoder(x=batch2.x, edge_index=batch2.edge_index, edge_weight=edge_weight)

        s = self.readout(h1, keepdim=True)
        s = self.sigmoid(s)
        return s, h1, h2

    def get_data_pairs(self, s, h1, h2):
        s = s.expand_as(h1)
        pos_pairs = torch.stack([s, h1], -1)
        neg_pairs = torch.stack([s, h2], -1)
        return pos_pairs, neg_pairs
