import torch

from .base import Model
from src.nn.utils import AvgReadout

from torch_geometric.typing import Tensor, Adj, OptTensor


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

    def forward(self, c: Tensor, h_pl: Tensor, h_mi: Tensor, s_bias1: OptTensor = None, s_bias2: OptTensor = None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)
        return logits


class ModelDGI(Model):
    r"""The full model to train the encoder.

    Args:
        encoder (torch.nn.Module): the encoder to be trained.
        discriminator (torch.nn.Module): the discriminator for contrastive learning.
    """
    def __init__(self, encoder: torch.nn.Module, discriminator: torch.nn.Module = Discriminator()):
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
        embs = self.encoder(seq=x, adj=adj, is_sparse=is_sparse).detach()
        if is_numpy:
            return embs.cpu().numpy()
        return embs
