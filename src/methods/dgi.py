import torch

from src.augment.collections import augment_dgi
from .base import BaseMethod

from typing import Callable
from src.typing import Tensor, OptAugment


class Discriminator(torch.nn.Module):
    def __init__(self, dim_h: int = 512):
        super().__init__()
        self.f_k = torch.nn.Bilinear(dim_h, dim_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c: Tensor, h_pl: Tensor, h_mi: Tensor):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x))
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x))

        logits = torch.stack((sc_1, sc_2))
        return logits


class DGI(BaseMethod):
    r"""
    TODO: add descriptions
    """
    def __init__(self,
                 encoder: torch.nn.Module,
                 discriminator: torch.nn.Module = Discriminator(),
                 data_augment: OptAugment = augment_dgi,
                 loss_function: Callable = torch.nn.BCEWithLogitsLoss()):
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


def get_label_pairs(n_pos: int, n_neg: int):
    r"""Get the positive and negative files."""
    label_pos = torch.ones(n_pos)
    label_neg = torch.zeros(n_neg)
    labels = torch.stack((label_pos, label_neg))
    return labels
