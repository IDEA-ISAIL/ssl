import os
import os.path as osp

from typing import Callable, List, Optional

import numpy as np
import scipy.sparse as sp
import torch

from torch_geometric.data import HeteroData, InMemoryDataset



def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




class FreebaseMovies(InMemoryDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self) -> List[str]:
        return ['labels.npy', 'ma.txt', 'mam.npz', 'md.txt', 'mdm.npz', 'mw.txt', 'mwm.npz', 'nei_a.npy', 'nei_d.npy', 'nei_w.npy', 'pos.npz']
    
    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    

    def process(self):
        data = HeteroData()

        node_types = ['movie', 'director', 'actor', 'writer']
        type_num = [3492, 2502, 33401, 4459]

        # features
        # data['movie'].x = torch.FloatTensor(preprocess_features(sp.eye(type_num[0])))
        # data['director'].x = torch.FloatTensor(preprocess_features(sp.eye(type_num[1])))
        # data['actor'].x = torch.FloatTensor(preprocess_features(sp.eye(type_num[2])))
        # data['writer'].x = torch.FloatTensor(preprocess_features(sp.eye(type_num[3])))
        # data['feats'] = [data['movie'].x.cuda(), data['director'].x.cuda(), data['actor'].x.cuda(), data['writer'].x.cuda()]

        data['movie'].x = torch.FloatTensor(preprocess_features(sp.eye(type_num[0])))
        feat1 = torch.FloatTensor(preprocess_features(sp.eye(type_num[1])))
        feat2 = torch.FloatTensor(preprocess_features(sp.eye(type_num[2])))
        feat3 = torch.FloatTensor(preprocess_features(sp.eye(type_num[3])))
        data['feats'] = [data['movie'].x.cuda(), feat1.cuda(), feat2.cuda(), feat3.cuda()]

        # mps
        mam = sp.load_npz(osp.join(self.raw_dir, "mam.npz"))
        mdm = sp.load_npz(osp.join(self.raw_dir, "mdm.npz"))
        mwm = sp.load_npz(osp.join(self.raw_dir, "mwm.npz"))
        
        data['mps'] = [sparse_mx_to_torch_sparse_tensor(normalize_adj(mam)).cuda(), sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm)).cuda(), sparse_mx_to_torch_sparse_tensor(normalize_adj(mwm)).cuda()]        


        # pos
        pos = sp.load_npz(osp.join(self.raw_dir, "pos.npz"))
        data['pos'] = sparse_mx_to_torch_sparse_tensor(pos).cuda()

        # nei_index
        nei_a = np.load(osp.join(self.raw_dir, "nei_a.npy"), allow_pickle=True)
        nei_d = np.load(osp.join(self.raw_dir, "nei_d.npy"), allow_pickle=True)
        nei_w = np.load(osp.join(self.raw_dir, "nei_w.npy"), allow_pickle=True)
        nei_a = [torch.LongTensor(i) for i in nei_a]
        nei_d = [torch.LongTensor(i) for i in nei_d]
        nei_w = [torch.LongTensor(i) for i in nei_w]
        data['nei_index'] = [nei_a, nei_d, nei_w]


        y = np.load(osp.join(self.raw_dir, 'labels.npy'))
        data['paper'].y = torch.from_numpy(y).to(torch.long)


        torch.save(self.collate([data]), self.processed_paths[0])


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'