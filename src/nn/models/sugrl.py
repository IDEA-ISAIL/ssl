import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from .base import BaseModel
from torch_geometric.typing import Tensor, Adj
from torch_geometric.nn.models import GCN

class SugrlModel(BaseModel):
    r"""The full model to train the encoder.

    Args:
        encoder (torch.nn.Module): the encoder to be trained.
        discriminator (torch.nn.Module): the discriminator for contrastive learning.
    """
    def __init__(self, encoder: torch.nn.Module):
        super().__init__(encoder=encoder)
        self.encoder_1 = encoder[0]
        self.encoder_2 = encoder[1]
        # super(SUGRL, self).__init__()
        # for m in self.modules():
        #     self.weights_init(m)

    def forward(self, x: Tensor, adj: Adj, is_sparse: bool = True):
        h_a = self.encoder_1(x)
        h_p = self.encoder_2(x, adj, is_sparse)
        return h_a, h_p

    def get_embs(self, x: Tensor, adj: Adj, is_sparse: bool = True):
        h_a, h_p = self.forward(x, adj, is_sparse)
        return h_a.detach(), h_p.detach()

    def get_embs_numpy(self, x: Tensor, adj: Adj, is_sparse: bool = True):
        embs = self.get_embs(x=x, adj=adj, is_sparse=is_sparse)
        return embs.cpu().to_numpy()

class SugrlMLP(nn.Module):
    def __init__(self, in_channels,dropout=0.2, cfg=[512, 128], batch_norm=False, out_layer=None):
        super(SugrlMLP, self).__init__()
        self.in_channels = in_channels
        self.layers = nn.ModuleList()
        self.layer_num = len(cfg)
        self.dropout = dropout
        for i in range(self.layer_num):
            if batch_norm:
                self.layers.append(nn.Linear(self.in_channels, cfg[i]))
                self.layers.append(nn.BatchNorm1d(self.out_channels, affine=False))
                self.layers.append(nn.ReLU())
            elif i != (self.layer_num-1):
                self.layers.append(nn.ReLU())
            else:
                 self.layers.append(nn.Linear(self.in_channels, cfg[i]))
            self.out_channels = cfg[i]

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        # print(self.layers)
        for i, l in enumerate(self.layers):
            # print(i)
            x = l(x)

        return x

class SugrlGCN(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 dim_out: int = 128,
                 act: torch.nn = torch.nn.PReLU(),
                 bias: bool = False):
        super(SugrlGCN, self).__init__()
        self.dim_out = dim_out
        self.fc = torch.nn.Linear(in_channels, dim_out, bias=False).to("cuda:0")
        self.act = act

        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(dim_out))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self._weights_init(m)

    def _weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, is_sparse=True):
        
        seq_fts = self.fc(seq)
        print("fc",self.fc)
        print("seq",seq)
        print("seq_fts",seq_fts)
        if is_sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
            print("out",torch.squeeze(seq_fts, 0))
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return out

# class GCN(nn.Module):
#     def __init__(self, dropout=0.2):
#         self.sparse = True
#         self.dropout = dropout
#         self.A = None

#     def forward(self, x, adj=None, sparse=True):
#         self.sparse = sparse
#         if self.A is None:
#             self.A = adj
#         x = F.dropout(x, 0.2, training=self.training)
#         if self.sparse:
#             x = torch.spmm(self.A, x)
#         else:
#             x = torch.mm(self.A, x)
#         return x

# class SugrlGCN(torch.nn.Module):
#     def __init__(self,
#                  in_channels: int,
#                  hidden_channels: int = 512,
#                  num_layers=1):
#         super(SugrlGCN, self).__init__()
#         # self.dim_out = hidden_channels
#         # self.fc = torch.nn.Linear(in_channels, hidden_channels, bias=False)
#         # self.act = act
#         #
#         # if bias:
#         #     self.bias = torch.nn.Parameter(torch.FloatTensor(hidden_channels))
#         #     self.bias.data.fill_(0.0)
#         # else:
#         #     self.register_parameter('bias', None)
#         #
#         # for m in self.modules():
#         #     self._weights_init(m)
#         # self.fc = torch.nn.Linear(in_channels, hidden_channels, bias=False)
#         self.gcn = GCN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers)
#         # self.act = act
#         for m in self.modules():
#             self._weights_init(m)

#     def _weights_init(self, m):
#         if isinstance(m, torch.nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)

#     def forward(self, batch, edge_index, is_sparse=True):
#         edge_weight = None
#         # edge_weight = batch.edge_weight #if "edge_weight" in batch else None
#         return self.gcn(x=batch.x, edge_index=edge_index, edge_weight=edge_weight)

