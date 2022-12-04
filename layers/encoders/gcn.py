import torch
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    r"""
    #TODO: need to add descriptions.
    """
    def __init__(
        self, 
        n_feat: int,
        d_hid: int,
        n_class: int,
        dropout: float = 0.5,
        n_layers: int = 1,
        normalize: bool = False
    ):

        super().__init__()
        
        self.num_layers = n_layers
        self.dropout = dropout
        self.input_fc = GCNConv(n_feat, d_hid)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(GCNConv(d_hid, d_hid))
            if normalize:
                self.bns.append(nn.BatchNorm1d(d_hid))
        self.out_fc = nn.Linear(d_hid, n_class)
        self.reset_parameters()
        self.normalize = normalize
        self.activation = nn.ReLU(True)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.input_fc.reset_parameters()
        self.out_fc.reset_parameters()

    def forward(self, data) -> Tensor:
        x, adj_t = data.x, data.adj_t
        h = self.input_fc(x, adj_t)
        for i in range(self.num_layers):
            h = self.activation(self.convs[i](F.dropout(h, p=self.dropout, training=self.training), adj_t))
            if self.normalize:
                h = self.bns[i](h)
        out = self.out_fc(h)
        out = F.log_softmax(out, dim=1)
        return out


class GCN_DGI(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN_DGI, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)