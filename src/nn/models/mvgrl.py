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

    def forward(self, c_1: Tensor, c_2: Tensor, h_1: Tensor, h_2: Tensor, h_3: Tensor, h_4: Tensor,
                s_bias1: OptTensor = None, s_bias2: OptTensor = None):
        c_x1 = torch.unsqueeze(c_1, 1)
        c_x1 = c_x1.expand_as(h_1).contiguous()

        c_x2 = torch.unsqueeze(c_2, 1)
        c_x2 = c_x2.expand_as(h_2).contiguous()

        sc_1 = torch.squeeze(self.f_k(h_2, c_x1), 2)
        sc_2 = torch.squeeze(self.f_k(h_1, c_x2), 2)

        sc_3 = torch.squeeze(self.f_k(h_4, c_x1), 2)
        sc_4 = torch.squeeze(self.f_k(h_3, c_x2), 2)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)
        return logits


class ModelMVGRL(Model):
    r"""The full model to train the encoder.

    Args:
        encoder (torch.nn.Module): the encoder to be trained.
        discriminator (torch.nn.Module): the discriminator for contrastive learning.
    """
    def __init__(self, encoder: torch.nn.Module, discriminator: torch.nn.Module = Discriminator()):
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
