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




class ACM(InMemoryDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self) -> List[str]:
        return ['a_feat.npz', 'labels.npy', 'nei_a.npy', 'nei_s.npy', 'p_feat.npz', 'pa.txt', 'pap.npz', 'pos.npz', 'ps.txt', 'psp.npz']
    
    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    

    def process(self):
        data = HeteroData()

        node_types = ['paper', 'author', 'subject']
        type_num = [4019, 7167, 60]

        # features
        data['paper'].x = torch.FloatTensor(preprocess_features(sp.load_npz(osp.join(self.raw_dir, 'p_feat.npz'))))
        data['author'].x = torch.FloatTensor(preprocess_features(sp.eye(type_num[1])))
        data['subject'].x = torch.FloatTensor(preprocess_features(sp.eye(type_num[2])))
        data['feats'] = [data['paper'].x.cuda(), data['author'].x.cuda(), data['subject'].x.cuda()]

        # mps
        pap = sp.load_npz(osp.join(self.raw_dir, "pap.npz"))
        psp = sp.load_npz(osp.join(self.raw_dir, "psp.npz"))
        
        data['mps'] = [sparse_mx_to_torch_sparse_tensor(normalize_adj(pap)).cuda(), sparse_mx_to_torch_sparse_tensor(normalize_adj(psp)).cuda()]        


        # pos
        pos = sp.load_npz(osp.join(self.raw_dir, "pos.npz"))
        data['pos'] = sparse_mx_to_torch_sparse_tensor(pos).cuda()

        # nei_index
        nei_a = np.load(osp.join(self.raw_dir, "nei_a.npy"), allow_pickle=True)
        nei_s = np.load(osp.join(self.raw_dir, "nei_s.npy"), allow_pickle=True)
        nei_a = [torch.LongTensor(i) for i in nei_a]
        nei_s = [torch.LongTensor(i) for i in nei_s]
        data['nei_index'] = [nei_a, nei_s]


        y = np.load(osp.join(self.raw_dir, 'labels.npy'))
        data['paper'].y = torch.from_numpy(y).to(torch.long)


        torch.save(self.collate([data]), self.processed_paths[0])


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'