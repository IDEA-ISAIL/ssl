import torch

from torch_geometric.typing import Tensor


class AvgReadout(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, keepdim: bool = False):
        return torch.mean(x, -2, keepdim=keepdim)

