import copy
import torch
from .base import Augmentor
from data import Data, HomoData
import random
import numpy as np
import scipy.sparse as sp


def delete_row_col(input_matrix, drop_list, only_row=False):
    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]
    return out


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

        # single_index_list = []
        # for i in list(index_list):
        #     single_index_list.append(i)
        #     index_list.remove((i[1], i[0]))

        edge_num = int(len(row_idx) / 2)  # 9228 / 2
        add_drop_num = int(edge_num * percent / 2)
        aug_adj = copy.deepcopy(data_tmp.adj.to_dense())

        edge_idx = [i for i in range(edge_num)]
        drop_idx = random.sample(edge_idx, add_drop_num)

        for i in drop_idx:
            aug_adj[index_list[i][0]][index_list[i][1]] = 0
            aug_adj[index_list[i][1]][index_list[i][0]] = 0

        '''
        above finish drop edges
        '''
        node_num = data_tmp.x.shape[0]
        l = [(i, j) for i in range(node_num) for j in range(i)]
        add_list = random.sample(l, add_drop_num)

        for i in add_list:
            aug_adj[i[0]][i[1]] = 1
            aug_adj[i[1]][i[0]] = 1

        # aug_adj = np.matrix(aug_adj)
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

        drop_node_list = sorted([i for i in all_node_list if not i in sub_node_id_list])

        aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
        aug_input_adj = delete_row_col(input_adj, drop_node_list)

        data_tmp.x = aug_input_fea
        data_tmp.adj = aug_input_adj.to_sparse()

        return data_tmp





