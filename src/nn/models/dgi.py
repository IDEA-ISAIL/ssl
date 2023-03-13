import torch

from .base import BaseModel
from src.nn.utils import AvgReadout

from torch_geometric.typing import Tensor, Adj, OptTensor


class Encoder(torch.nn.Module):
    """
    This encoder is GCN.
    """
    def __init__(self,
                 dim_in: int,
                 dim_out: int = 512,
                 act: torch.nn = torch.nn.PReLU(),
                 bias: bool = True):
        super(Encoder, self).__init__()
        self.dim_out = dim_out
        self.fc = torch.nn.Linear(dim_in, dim_out, bias=False)
        self.act = act

        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(dim_out))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self._weights_init(m)

    def _weights_init(self, m):
        """
        TODO: maybe move to utils.
        """
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, is_sparse=False):
        seq_fts = self.fc(seq)
        if is_sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


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

        # sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        # sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)
        # torch_geometric
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x))
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x))

        # logits = torch.cat((sc_1, sc_2), 1)
        logits = torch.stack((sc_1, sc_2))
        return logits


class Model(BaseModel):
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

    # def forward(self, x: Tensor, x_neg: Tensor, adj: Adj, is_sparse: bool = True):
    #     h_1 = self.encoder(x, adj, is_sparse)
    #
    #     c = self.read(h_1)
    #     c = self.sigmoid(c)
    #
    #     h_2 = self.encoder(x_neg, adj, is_sparse)
    #
    #     logits = self.discriminator(c, h_1, h_2)
    #     return logits

    # torch_geometric gcn
    def forward(self, data, data_neg):
        h_1 = self.encoder(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight)

        c = self.read(h_1)
        c = self.sigmoid(c)

        h_2 = self.encoder(x=data_neg.x, edge_index=data_neg.edge_index, edge_weight=data_neg.edge_weight)

        logits = self.discriminator(c, h_1, h_2)
        return logits

    def get_embs(self, x: Tensor, adj: Adj, is_sparse: bool = True, is_numpy: bool = False):
        embs = self.encoder(seq=x, adj=adj, is_sparse=is_sparse).detach()
        if is_numpy:
            return embs.cpu().numpy()
        return embs
