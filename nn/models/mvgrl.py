import torch

from .base import Model
from nn.utils import AvgReadout, DiscriminatorMVGRL

from torch_geometric.typing import Tensor, Adj


class ModelMVGRL(Model):
    r"""The full model to train the encoder.

    Args:
        encoder (torch.nn.Module): the encoder to be trained.
        discriminator (torch.nn.Module): the discriminator for contrastive learning.
    """
    def __init__(self, encoder: torch.nn.Module, discriminator: torch.nn.Module = DiscriminatorMVGRL()):
        super().__init__(encoder=encoder)
        self.encoder_1 = encoder[0]
        self.encoder_2 = encoder[1]
        self.discriminator = discriminator
        self.read = AvgReadout()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: Tensor, x_neg: Tensor, adj: Adj, diff: Adj, is_sparse: bool = True,
                msk: Tensor = None, samp_bias1: Tensor = None, samp_bias2: Tensor = None):
        h_1 = self.encoder_1(x, adj, is_sparse)
        c_1 = self.read(h_1, msk)
        c_1 = self.sigmoid(c_1)

        h_2 = self.encoder_2(x, diff, is_sparse)
        c_2 = self.read(h_2, msk)
        c_2 = self.sigmoid(c_2)

        h_3 = self.encoder_1(x_neg, adj, is_sparse)

        h_4 = self.encoder_2(x_neg, diff, is_sparse)

        ret = self.discriminator(c_1, c_2, h_1, h_2, h_3, h_4, samp_bias1, samp_bias2)
        return ret, h_1, h_2

    def get_embs(self, x: Tensor, adj: Adj, diff: Adj, is_sparse: bool = True, msk: Tensor = None):
        h_1 = self.encoder_1(x, adj, is_sparse)
        c_1 = self.read(h_1, msk)

        h_2 = self.encoder_2(x, diff, is_sparse)
        return (h_1 + h_2).detach(), c_1.detach()

    def get_embs_numpy(self, x: Tensor, adj: Adj, diff: Adj, is_sparse: bool = True, msk: Tensor = None):
        embs = self.get_embs(x=x, adj=adj, diff=diff, is_sparse=is_sparse)
        return embs.cpu().to_numpy()
