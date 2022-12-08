import os
import sys
import pickle as pkl
import networkx as nx

from data.base_data import HomoData
from data.utils import *


class DGIData(HomoData):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.idx_train = None
        self.idx_val = None
        self.idx_test = None

    def load(self, path):  # {'pubmed', 'citeseer', 'cora'}
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

        self.attrs = sp.vstack((allx, tx)).tolil()
        self.attrs[test_idx_reorder, :] = self.attrs[test_idx_range, :]
        self.adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        self.labels = np.vstack((ally, ty))
        self.labels[test_idx_reorder, :] = self.labels[test_idx_range, :]

        self.idx_test = test_idx_range.tolist()
        self.idx_train = range(len(y))
        self.idx_val = range(len(y), len(y) + 500)


data = DGIData()
data.load(path="../datasets/cora_dgi")
