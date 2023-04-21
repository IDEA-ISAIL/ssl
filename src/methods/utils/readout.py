import torch
from torch_geometric.typing import Tensor


class AvgReadout(object):
    def __init__(self) -> None:
        return

    def __call__(self, x: Tensor, keepdim: bool = False):
        return torch.mean(x, -2, keepdim=keepdim)