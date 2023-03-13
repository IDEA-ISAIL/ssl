import copy
import random
from scipy.linalg import fractional_matrix_power

import torch
from torch.linalg import inv
from torch_geometric.data import Data

from .base import Augmentor
from src.data import HomoData


class DataShuffle(Augmentor):
    """TODO: to be removed"""
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


class NodeShuffle(Augmentor):
    def __init__(self):
        super().__init__()

    def apply(self, data: Data):
        data_tmp = copy.deepcopy(data)
        idx_tmp = torch.randperm(len(data.x))
        data_tmp.x = data.x[idx_tmp]
        return data_tmp


class ComputePPR(Augmentor):
    def __init__(self, alpha=0.2, self_loop=True):
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
    def __init__(self, t=5, self_loop=True):
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


def delete_row_col(input_matrix, drop_list, only_row=False):
    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]
    return out


class RandomMask(Augmentor):
    def __init__(self, is_x: bool = True, is_adj: bool = False):
        super().__init__()
        self.is_x = is_x
        self.is_adj = is_adj

    def apply(self, data: HomoData, drop_percent=0.2):
        data_tmp = copy.deepcopy(data)
        if self.is_x:
            mask_num = int(data_tmp.n_nodes * drop_percent)
            node_idx = [i for i in range(data_tmp.n_nodes)]
            mask_idx = random.sample(node_idx, mask_num)
            zeros = torch.zeros_like(data_tmp.x[0])
            for j in mask_idx:
                data_tmp.x[j, :] = zeros
        if self.is_adj:
            raise NotImplementedError
        return data_tmp


class RandomDropEdge(Augmentor):
    def __init__(self, is_x: bool = True, is_adj: bool = False):
        super().__init__()
        self.is_x = is_x
        self.is_adj = is_adj

    def apply(self, data: HomoData, drop_percent=0.2):
        data_tmp = copy.deepcopy(data)
        percent = drop_percent / 2
        row_idx, col_idx = data.x.nonzero().T

        index_list = []
        for i in range(len(row_idx)):
            index_list.append((row_idx[i], col_idx[i]))

        edge_num = int(len(row_idx) / 2)  # 9228 / 2
        add_drop_num = int(edge_num * percent / 2)
        aug_adj = copy.deepcopy(data_tmp.adj.to_dense())

        edge_idx = [i for i in range(edge_num)]
        drop_idx = random.sample(edge_idx, add_drop_num)

        for i in drop_idx:
            aug_adj[index_list[i][0]][index_list[i][1]] = 0
            aug_adj[index_list[i][1]][index_list[i][0]] = 0

        node_num = data_tmp.x.shape[0]
        l = [(i, j) for i in range(node_num) for j in range(i)]
        add_list = random.sample(l, add_drop_num)

        for i in add_list:
            aug_adj[i[0]][i[1]] = 1
            aug_adj[i[1]][i[0]] = 1

        data_tmp.adj = aug_adj.to_sparse()
        return data_tmp


class RandomDropNode(Augmentor):
    def __init__(self, is_x: bool = True, is_adj: bool = False):
        super().__init__()
        self.is_x = is_x
        self.is_adj = is_adj

    def apply(self, data: HomoData, drop_percent=0.2):
        data_tmp = copy.deepcopy(data)
        input_adj = torch.tensor(data_tmp.adj.to_dense().tolist())
        input_fea = data_tmp.x.squeeze(0)

        node_num = input_fea.shape[0]
        drop_num = int(node_num * drop_percent)  # number of drop nodes
        all_node_list = [i for i in range(node_num)]

        drop_node_list = sorted(random.sample(all_node_list, drop_num))

        aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
        aug_input_adj = delete_row_col(input_adj, drop_node_list)

        data_tmp.x = aug_input_fea
        data_tmp.adj = aug_input_adj.to_sparse()

        return data_tmp


class AugmentSubgraph(Augmentor):
    def __init__(self, is_x: bool = True, is_adj: bool = False):
        super().__init__()
        self.is_x = is_x
        self.is_adj = is_adj

    def apply(self, data: HomoData, drop_percent=0.2):
        data_tmp = copy.deepcopy(data)
        input_adj = data_tmp.adj.to_dense()
        input_fea = data_tmp.x.squeeze(0)
        node_num = input_fea.shape[0]

        all_node_list = [i for i in range(node_num)]
        s_node_num = int(node_num * (1 - drop_percent))
        center_node_id = random.randint(0, node_num - 1)
        sub_node_id_list = [center_node_id]
        all_neighbor_list = []

        for i in range(s_node_num - 1):

            all_neighbor_list += torch.nonzero(input_adj[sub_node_id_list[i]], as_tuple=False).squeeze(1).tolist()

            all_neighbor_list = list(set(all_neighbor_list))
            new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]
            if len(new_neighbor_list) != 0:
                new_node = random.sample(new_neighbor_list, 1)[0]
                sub_node_id_list.append(new_node)
            else:
                break

        drop_node_list = sorted([i for i in all_node_list if i not in sub_node_id_list])

        aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
        aug_input_adj = delete_row_col(input_adj, drop_node_list)

        data_tmp.x = aug_input_fea
        data_tmp.adj = aug_input_adj.to_sparse()

        return data_tmp
