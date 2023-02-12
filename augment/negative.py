import copy
import torch
from scipy.linalg import fractional_matrix_power
from torch.linalg import inv

from .base import Augmentor
from data import Data, HomoData


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
    
class ComputePPR(Augmentor):
    def __init__(self, alpha = 0.2, self_loop = True):
        super().__init__()
        self.alpha = alpha
        self.self_loop = self_loop

    def apply(self, data: HomoData):
        data_tmp = copy.deepcopy(data)
        a = data_tmp.adj
        if self.self_loop:
            a = torch.eye(a.shape[0]) + a
        d = torch.diag(torch.sum(a, 1))
        dinv = torch.from_numpy(fractional_matrix_power(d, -0.5))
        at = torch.matmul(torch.matmul(dinv, a), dinv)
        data_tmp.adj = self.alpha * inv((torch.eye(a.shape[0]) - (1 - self.alpha) * at))
        return data_tmp
    
class ComputeHeat(Augmentor):
    def __init__(self, t = 5, self_loop = True):
        super().__init__()
        self.t = t
        self.self_loop = self_loop

    def apply(self, data: HomoData):
        data_tmp = copy.deepcopy(data)
        a = data_tmp.adj
        if self.self_loop:
            a = torch.eye(a.shape[0]) + a
        d = torch.diag(torch.sum(a, 1))
        data_tmp.adj = torch.exp(self.t * (torch.matmul(a, inv(d)) - 1))
        return data_tmp
