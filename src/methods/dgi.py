import torch

from .base import BaseMethod
from src.augment import ShuffleNode
from .utils import get_label_pairs

from typing import Callable
from src.typing import OptAugment


class DGI(BaseMethod):
    r"""
    TODO: add descriptions
    """
    def __init__(self,
                 encoder: torch.nn.Module,
                 discriminator: torch.nn.Module,
                 data_augment: OptAugment = ShuffleNode,
                 loss_function: Callable = torch.nn.BCEWithLogitsLoss()) -> None:
        super().__init__(encoder=encoder,
                         data_augment=data_augment,
                         loss_function=loss_function)

        self.discriminator = discriminator

    def forward(self, batch, batch_neg):
        h_1 = self.encoder(x=batch.x, edge_index=batch.edge_index, edge_weight=batch.edge_weight)
        c = self.read(h_1)
        c = self.sigmoid(c)

        h_2 = self.encoder(x=batch_neg.x, edge_index=batch_neg.edge_index, edge_weight=batch_neg.edge_weight)

        logits = self.discriminator(c, h_1, h_2)
        return logits

    def train_iter(self, batch):
        device = batch.device
        self.model.train()
        batch_neg = self.data_augment(batch).to(device)

        logits = self.forward(batch=batch, batch_neg=batch_neg)
        labels = get_label_pairs(n_pos=batch.num_nodes, n_neg=batch.num_nodes).to(device)

        loss = self.loss_function(logits, labels)
        return loss
