import copy
import torch

from .base import Augmentor
from ssl.data import HomoData


class Shuffle(Augmentor):
    def __init__(self, is_x: bool = True, is_adj: bool = False):
        super().__init__()
        self.is_x = is_x
        self.is_adj = is_adj

    def apply(self, data: HomoData):
        data_tmp = copy.deepcopy(data)
        if self.is_x:
            idx_tmp = torch.randperm(data_tmp.n_nodes)
            data_tmp.x = data_tmp.x[idx_tmp]
        if self.is_adj:
            raise NotImplementedError
        return data_tmp
