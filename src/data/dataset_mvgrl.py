import os
import sys
import pickle as pkl
import networkx as nx
import torch_sparse

from .data import HomoData
from .dataset import Dataset
from .utils import *


class DatasetMVGRL(Dataset):
    def __init__(self):
        self.x = None
        self.adj = None
        self.labels = None
        self.idx_train = None
        self.idx_val = None
        self.idx_test = None

    def load(self, path):  # {'pubmed', 'citeseer', 'cora'}
        print("Loading data from {}".format(path))
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            p = os.path.join(path, names[i])
            with open(p, 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        p = os.path.join(path, "test.index")
        test_idx_reorder = parse_index_file(p)
        test_idx_range = np.sort(test_idx_reorder)

        # todo: take citeseer into consideration
        # if dataset_str == 'citeseer':
        #     # Fix citeseer dataset (there are some isolated nodes in the graph)
        #     # Find isolated nodes, add them as zero-vecs into the right position
        #     test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        #     tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        #     tx_extended[test_idx_range - min(test_idx_range), :] = tx
        #     tx = tx_extended
        #     ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        #     ty_extended[test_idx_range - min(test_idx_range), :] = ty
        #     ty = ty_extended

        self.x = sp.vstack((allx, tx)).tolil()
        self.x[test_idx_reorder, :] = self.x[test_idx_range, :]
        self.adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        self.x, _ = preprocess_features(self.x)
        self.adj = normalize_adj(self.adj + sp.eye(self.adj.shape[0]))

        self.x = torch.FloatTensor(self.x)
        self.adj = sparse_mx_to_torch_sparse_tensor(self.adj)
        # self.adj = torch.FloatTensor(self.adj)

        self.labels = np.vstack((ally, ty))
        self.labels[test_idx_reorder, :] = self.labels[test_idx_range, :]

        self.idx_test = test_idx_range.tolist()
        self.idx_train = range(len(y))
        self.idx_val = range(len(y), len(y) + 500)

    def to_data(self):
        return HomoData(x=self.x, adj=self.adj)

