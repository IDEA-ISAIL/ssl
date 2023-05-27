"""TODO: create a separate file for each class."""
import copy
import random
from scipy.linalg import fractional_matrix_power
import torch
from torch.linalg import inv
from torch_geometric.data import Data
from .base import Augmentor
from src.data import HomoData


class ComputePPR(Augmentor):
    def __init__(self, alpha=0.2, self_loop=True):
        super().__init__()
        self.alpha = alpha
        self.self_loop = self_loop

    def __call__(self, data: HomoData):
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

    def __call__(self, data: HomoData):
        data_tmp = copy.deepcopy(data)
        a = data_tmp.adj
        if self.self_loop:
            a = torch.eye(a.shape[0]) + a
        d = torch.diag(torch.sum(a, 1))
        data_tmp.adj = torch.exp(self.t * (torch.matmul(a, inv(d)) - 1))
        return data_tmp


# import faiss
import torch
import numpy as np
class NeighborSearch_AFGRL(Augmentor):
    def __init__(self, device="cuda", num_centroids=20, num_kmeans=5, clus_num_iters=20):
        super(NeighborSearch_AFGRL, self).__init__()
        self.device = device
        self.num_centroids = num_centroids
        self.num_kmeans = num_kmeans
        self.clus_num_iters = clus_num_iters

    def __get_close_nei_in_back(self, indices, each_k_idx, cluster_labels, back_nei_idxs, k):
        # get which neighbors are close in the background set
        batch_labels = cluster_labels[each_k_idx][indices]
        top_cluster_labels = cluster_labels[each_k_idx][back_nei_idxs]
        batch_labels = self.repeat_1d_tensor(batch_labels, k)

        curr_close_nei = torch.eq(batch_labels, top_cluster_labels)
        return curr_close_nei
    
    def repeat_1d_tensor(self, t, num_reps):
        return t.unsqueeze(1).expand(-1, num_reps)

    def __call__(self, adj, student, teacher, top_k):
        n_data, d = student.shape
        similarity = torch.matmul(student, torch.transpose(teacher, 1, 0).detach())
        similarity += torch.eye(n_data, device=self.device) * 10

        _, I_knn = similarity.topk(k=top_k, dim=1, largest=True, sorted=True)
        tmp = torch.LongTensor(np.arange(n_data)).unsqueeze(-1).to(self.device)
        knn_neighbor = self.create_sparse(I_knn)
        locality = knn_neighbor * adj

        ncentroids = self.num_centroids
        niter = self.clus_num_iters

        pred_labels = []

        for seed in range(self.num_kmeans):
            kmeans = faiss.Kmeans(d, ncentroids, niter=niter, gpu=False, seed=seed + 1234)
            kmeans.train(teacher.cpu().numpy())
            _, I_kmeans = kmeans.index.search(teacher.cpu().numpy(), 1)
        
            clust_labels = I_kmeans[:,0]

            pred_labels.append(clust_labels)

        pred_labels = np.stack(pred_labels, axis=0)
        cluster_labels = torch.from_numpy(pred_labels).long()

        all_close_nei_in_back = None
        with torch.no_grad():
            for each_k_idx in range(self.num_kmeans):
                curr_close_nei = self.__get_close_nei_in_back(tmp.squeeze(-1), each_k_idx, cluster_labels, I_knn, I_knn.shape[1])

                if all_close_nei_in_back is None:
                    all_close_nei_in_back = curr_close_nei
                else:
                    all_close_nei_in_back = all_close_nei_in_back | curr_close_nei

        all_close_nei_in_back = all_close_nei_in_back.to(self.device)

        globality = self.create_sparse_revised(I_knn, all_close_nei_in_back)

        pos_ = locality + globality

        return pos_.coalesce()._indices(), I_knn.shape[1]

    def create_sparse(self, I):
        
        similar = I.reshape(-1).tolist()
        index = np.repeat(range(I.shape[0]), I.shape[1])
        
        assert len(similar) == len(index)
        indices = torch.tensor([index, similar]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)), [I.shape[0], I.shape[0]], dtype=torch.float).to(self.device)

        return result

    def create_sparse_revised(self, I, all_close_nei_in_back):
        n_data, k = I.shape[0], I.shape[1]

        index = []
        similar = []
        for j in range(I.shape[0]):
            for i in range(k):
                index.append(int(j))
                similar.append(I[j][i].item())

        index = torch.masked_select(torch.LongTensor(index).to(self.device), all_close_nei_in_back.reshape(-1))
        similar = torch.masked_select(torch.LongTensor(similar).to(self.device), all_close_nei_in_back.reshape(-1))

        assert len(similar) == len(index)
        indices = torch.tensor([index.cpu().numpy().tolist(), similar.cpu().numpy().tolist()]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones(len(index)).to(self.device), [n_data, n_data], dtype=torch.float).to(self.device)

        return result
