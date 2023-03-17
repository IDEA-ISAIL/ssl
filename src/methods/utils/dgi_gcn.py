import torch


class DGIGCN(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 512,
                 act: torch.nn = torch.nn.PReLU(),
                 bias: bool = True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.fc = torch.nn.Linear(in_channels, hidden_channels, bias=False)
        self.act = act

        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(hidden_channels))
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

    def forward(self, x, adj, is_sparse=True):
        """TODO: check later for molecure graph."""
        x_fts = self.fc(x)
        if is_sparse:
            # out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(x_fts, 0)), 0)
            out = torch.spmm(adj, torch.squeeze(x_fts, 0))
        else:
            out = torch.bmm(adj, x_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)
