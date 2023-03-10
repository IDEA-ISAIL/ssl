import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np
import copy
from .base import Model
from torch_geometric.typing import Tensor, Adj
from src.data.utils import sparse_mx_to_torch_sparse_tensor
import faiss


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

    
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val




class ModelAFGRL(Model):
    r"""The full model to train the encoder.

    Args:
        encoder (torch.nn.Module): the encoder to be trained.
        discriminator (torch.nn.Module): the discriminator for contrastive learning.
    """
    def __init__(self, student_encoder: torch.nn.Module, teacher_encoder: torch.nn.Module, adj_ori = None, topk=8):
        super().__init__(encoder=student_encoder)
        self.encoder = student_encoder
        self.teacher_encoder = teacher_encoder
        set_requires_grad(self.teacher_encoder, False)
        rep_dim = self.encoder.dim_out
        pred_hid = rep_dim*2
        self.student_predictor = nn.Sequential(nn.Linear(rep_dim, pred_hid), nn.PReLU(), nn.Linear(pred_hid, rep_dim))
        self.student_predictor.apply(init_weights)
        self.neighbor = Neighbor()
        self.topk = topk
        self.adj_ori = adj_ori

    def forward(self, x: Tensor, adj: Adj, is_sparse: bool = True):
        student = self.encoder(x, adj, is_sparse)
        pred = self.student_predictor(student)

        with torch.no_grad():
            teacher = self.teacher_encoder(x, adj, is_sparse)
    
        adj_search = self.adj_ori
        student, teacher, pred = torch.squeeze(student), torch.squeeze(teacher), torch.squeeze(pred)
        ind, k = self.neighbor(adj_search, F.normalize(student, dim=-1, p=2), F.normalize(teacher, dim=-1, p=2), self.topk)

        loss1 = loss_fn(pred[ind[0]], teacher[ind[1]].detach())
        loss2 = loss_fn(pred[ind[1]], teacher[ind[0]].detach())
        loss = loss1 + loss2

        return loss.mean()

    def get_embs(self, x: Tensor, adj: Adj, is_sparse: bool = True):
        embs = self.encoder(x=x, adj=adj, is_sparse=is_sparse)
        return embs.detach()

    def get_embs_numpy(self, x: Tensor, adj: Adj, is_sparse: bool = True):
        embs = self.get_embs(x=x, adj=adj, is_sparse=is_sparse)
        return embs.cpu().to_numpy()


def repeat_1d_tensor(t, num_reps):
    return t.unsqueeze(1).expand(-1, num_reps)

class Neighbor(nn.Module):
    def __init__(self, device="cuda", num_centroids=20, num_kmeans=5, clus_num_iters=20):
        super(Neighbor, self).__init__()
        self.device = device
        self.num_centroids = num_centroids
        self.num_kmeans = num_kmeans
        self.clus_num_iters = clus_num_iters

    def __get_close_nei_in_back(self, indices, each_k_idx, cluster_labels, back_nei_idxs, k):
        # get which neighbors are close in the background set
        batch_labels = cluster_labels[each_k_idx][indices]
        top_cluster_labels = cluster_labels[each_k_idx][back_nei_idxs]
        batch_labels = repeat_1d_tensor(batch_labels, k)

        curr_close_nei = torch.eq(batch_labels, top_cluster_labels)
        return curr_close_nei

    def forward(self, adj, student, teacher, top_k):
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