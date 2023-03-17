import torch
import torch.nn as nn

from torch_geometric.typing import Tensor


class LogReg(nn.Module):
    def __init__(self, dim_in: int, n_classes: int):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(dim_in, n_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: Tensor):
        ret = self.fc(x)
        return ret
