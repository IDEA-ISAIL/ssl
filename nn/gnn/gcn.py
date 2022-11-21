import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class GCN(nn.Module):
    def __init__(
        self, 
        n_feat, 
        d_hid, 
        n_class, 
        dropout=0.5, 
        n_layers=1, 
        normalize=False
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

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        h = self.input_fc(x, adj_t)
        for i in range(self.num_layers):
            h = self.activation(self.convs[i](F.dropout(h, p=self.dropout, training=self.training), adj_t))
            if self.normalize:
                h = self.bns[i](h)
        out = self.out_fc(h)
        out = F.log_softmax(out, dim=1)
        return out
