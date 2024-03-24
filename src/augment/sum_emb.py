import torch
from torch_geometric.typing import Tensor

from .base import Augmentor


class GlobalSum(Augmentor):
    def __init__(self, 
                 pooler: str="avg"):
        super().__init__()
        assert pooler in ["avg", "max"], 'Pooler should be either "avg" or "max"'
        self.pooler = pooler

    def __call__(self, 
                 x: Tensor, 
                 dim: int=1, 
                 keepdim: bool=False):
        if self.pooler == "avg":
            return torch.mean(x, dim, keepdim=keepdim)
        else:
            return torch.max(x, dim, keepdim=keepdim)
        
