import torch

from .base import BaseMethod
from src.augment import ShuffleNode
from .utils import get_label_pairs, AvgReadout

from typing import Callable, Union
from src.typing import AugmentType


CONV_TYPES = ['spectral', 'spatial', 0, 1]


class DGI(BaseMethod):
    r""" Deep Graph Infomax (DGI).

    Args:
        encoder (torch.nn.Module): the encoder to be trained.

        discriminator (torch.nn.Module): the discriminator to discriminate positive and negative node pairs.

        data_augment (AugmentType): data augmentation to generate negative node pairs. (default: ShuffleNode()).

        loss_function (Callable): the loss function. (default: torch.nn.BCEWithLogitsLoss())

        conv_type (Union[str, int]): 'spectral' (0) or 'spatial' (1).
            'spectral': spectral convolution.
                It updates node embeddings by AXW.
                The inputs are the feature matrix 'x' & adjacency matrix 'adj'.
            'spatial': spatial convolution.
                It updates node embeddings by message passing (the built-in GCN).
                The inputs are the feature matrix 'x' & 'edge_index' & 'edge_weight'.
            (default: 'spectral')
    """
    def __init__(self,
                 encoder: torch.nn.Module,
                 discriminator: torch.nn.Module,
                 data_augment: AugmentType = ShuffleNode(),
                 loss_function: Callable = torch.nn.BCEWithLogitsLoss(),
                 conv_type: Union[str, int] = 'spectral') -> None:
        super().__init__(encoder=encoder, data_augment=data_augment, loss_function=loss_function)

        self.discriminator = discriminator
        self.read = AvgReadout()
        self.sigmoid = torch.nn.Sigmoid()

        assert conv_type in CONV_TYPES
        self.conv_type = conv_type

    def forward(self, batch, batch_neg):
        h_1 = self.encoder_switch(batch)
        c = self.read(h_1)
        c = self.sigmoid(c)

        h_2 = self.encoder_switch(batch_neg)

        logits = self.discriminator(c, h_1, h_2)
        return logits

    def train_iter(self, batch):
        batch_neg = self.data_augment(batch).to(self._device)
        logits = self.forward(batch=batch, batch_neg=batch_neg)
        labels = get_label_pairs(n_pos=batch.num_nodes, n_neg=batch.num_nodes).to(self._device)
        loss = self.loss_function(logits, labels)
        return loss

    def encoder_switch(self, batch):
        if self.conv_type == "spectral" or self.conv_type == 0:
            h = self.encoder(x=batch.x, adj=batch.adj_t.to_torch_sparse_coo_tensor())
        if self.conv_type == "spatial" or self.conv_type == 1:
            edge_weight = batch.edge_weight if "edge_weight" in batch else None
            h = self.encoder(x=batch.x, edge_index=batch.edge_index, edge_weight=edge_weight)
        return h
