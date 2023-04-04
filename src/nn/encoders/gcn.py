"""
TODO: move method-specific encoders to their corresponding files.
"""

import torch
from torch import Tensor

import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
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
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(GCNConv(d_hid, d_hid))
            if normalize:
                self.bns.append(torch.nn.BatchNorm1d(d_hid))
        self.out_fc = torch.nn.Linear(d_hid, n_class)
        self.reset_parameters()
        self.normalize = normalize
        self.activation = torch.nn.ReLU(True)

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


    