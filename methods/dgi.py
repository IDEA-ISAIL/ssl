import torch
from nn import GCN_DGI, AvgReadout, DiscriminatorDGI
from methods import BaseMethod
from typing import Tuple, List, Dict, Any


class DGI_old(torch.nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI_old, self).__init__()
        self.gcn = GCN_DGI(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = torch.nn.Sigmoid()

        self.disc = DiscriminatorDGI(n_h)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()


class DGI(BaseMethod):
    def __init__(self,
                 encoder=GCN_DGI,
                 data_transform: Any,
                 data_iterator: Any,
                 discriminator: DiscriminatorDGI,
                 ):
        super().__init__(
            encoder=encoder,
            data_transform=data_transform,
            data_iterator=data_iterator
        )

        self.discriminator = discriminator
        self.read = AvgReadout()
        self.sigm = torch.nn.Sigmoid()

    def get_loss(self, **kwargs):
        pass

    def train(self):
        pass

