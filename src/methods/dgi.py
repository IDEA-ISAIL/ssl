import torch
from torch_geometric.nn.models import GCN

from .base import BaseMethod
from .utils import AvgReadout
from src.augment import ShuffleNode
from src.losses import NegativeMI

from typing import Optional
from src.typing import AugmentType


class DGI(BaseMethod):
    r""" Deep Graph Infomax (DGI).

    Args:
        in_channels (Optional[int]): the input channels of the encoder. Must be specified if use the default "encoder".
        hidden_channels (Optional[int]): the hidden channels of the encoder. Must be specified if use the default
            "encoder" and "loss_function".
        encoder (Optional[torch.nn.Module]): the encoder to be trained. If use the default encoder, must specify
            "in_channels" and "hidden_channels".
        data_augment (AugmentType): data augmentation to generate negative node pairs.
            (default: ShuffleNode).
        loss_function (Callable): the loss function.
            (default: NegativeMI)
    """
    def __init__(self,
                 in_channels: Optional[int] = None,
                 hidden_channels: Optional[int] = None,
                 encoder: Optional[torch.nn.Module] = None,
                 readout: Optional[torch.nn.Module] = None,
                 data_augment: AugmentType = ShuffleNode(),
                 loss_function: Optional[torch.nn.Module] = None) -> None:
        super().__init__(encoder=encoder, data_augment=data_augment, loss_function=loss_function)

        if self.encoder is None:
            assert in_channels > 0 and hidden_channels > 0,\
                "If use the default GCN encoder, then in_channels and hidden_channels must be set."
            self.encoder = Encoder(in_channels=in_channels, hidden_channels=hidden_channels)

        if self.loss_function is None:
            assert hidden_channels > 0, \
                "If use the default loss function (negative mutual information), then hidden_channels must be set."
            self.loss_function = NegativeMI(in_channels=hidden_channels)

        self.readout = readout if readout else AvgReadout()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, batch, batch_neg):
        edge_weight = batch.edge_weight if "edge_weight" in batch else None
        h_pos = self.encoder(x=batch.x, edge_index=batch.edge_index, edge_weight=edge_weight)
        h_neg = self.encoder(x=batch_neg.x, edge_index=batch.edge_index, edge_weight=edge_weight)

        s = self.readout(h_pos, keepdim=True)
        s = self.sigmoid(s)
        return s, h_pos, h_neg

    def train_iter(self, batch):
        batch_neg = self.data_augment(batch).to(self._device)
        s, h_pos, h_neg = self.forward(batch=batch, batch_neg=batch_neg)
        s = s.expand_as(h_pos)
        loss = self.loss_function(x=s, y=h_pos, x_ind=s, y_ind=h_neg)
        return loss


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=512, num_layers=1, act=torch.nn.PReLU()):
        super().__init__()
        self.gcn = GCN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, act=act)
        self.act = act

    def forward(self, x, edge_index, edge_weight=None):
        return self.act(self.gcn(x=x, edge_index=edge_index, edge_weight=edge_weight))
