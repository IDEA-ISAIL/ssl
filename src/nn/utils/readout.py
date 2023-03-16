import torch

from torch_geometric.typing import Tensor, OptTensor


class AvgReadout(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, msk: OptTensor = None):
        if msk is None:
            return torch.mean(x, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(x * msk, 1) / torch.sum(msk)
