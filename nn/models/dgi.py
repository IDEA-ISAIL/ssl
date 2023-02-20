import torch

from .base import Model
from nn.utils import AvgReadout, DiscriminatorDGI

from torch_geometric.typing import Tensor, Adj


class ModelDGI(Model):
    r"""The full model to train the encoder.

    Args:
        encoder (torch.nn.Module): the encoder to be trained.
        discriminator (torch.nn.Module): the discriminator for contrastive learning.
    """
    def __init__(self, encoder: torch.nn.Module, discriminator: torch.nn.Module = DiscriminatorDGI()):
        super().__init__(encoder=encoder)
        self.discriminator = discriminator
        self.read = AvgReadout()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: Tensor, x_neg: Tensor, adj: Adj, is_sparse: bool = True,
                msk: Tensor = None, samp_bias1: Tensor = None, samp_bias2: Tensor = None):
        h_1 = self.encoder(x, adj, is_sparse)

        c = self.read(h_1, msk)
        c = self.sigmoid(c)

        h_2 = self.encoder(x_neg, adj, is_sparse)

        logits = self.discriminator(c, h_1, h_2, samp_bias1, samp_bias2)
        return logits

    def get_embs(self, x: Tensor, adj: Adj, is_sparse: bool = True, is_numpy: bool = False):
        embs = self.encoder(x=x, adj=adj, is_sparse=is_sparse).detach()
        if is_numpy:
            return embs.cpu().to_numpy()
        return embs
