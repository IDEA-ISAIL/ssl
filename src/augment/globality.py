import faiss
import torch
import numpy as np
def cluster_global(teacher, I_knn, num_centroids, clus_num_iters, num_kmeans, device):
    n_data, d = teacher.shape
    ncentroids = num_centroids
    niter = clus_num_iters
    tmp = torch.LongTensor(np.arange(n_data)).unsqueeze(-1).to(device)
    pred_labels = []

    for seed in range(num_kmeans):
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, gpu=False, seed=seed + 1234)
        kmeans.train(teacher.cpu().numpy())
        _, I_kmeans = kmeans.index.search(teacher.cpu().numpy(), 1)
    
        clust_labels = I_kmeans[:,0]

        pred_labels.append(clust_labels)

    pred_labels = np.stack(pred_labels, axis=0)
    cluster_labels = torch.from_numpy(pred_labels).long()

    all_close_nei_in_back = None
    with torch.no_grad():
        for each_k_idx in range(num_kmeans):
            curr_close_nei = __get_close_nei_in_back(tmp.squeeze(-1), each_k_idx, cluster_labels, I_knn, I_knn.shape[1])

            if all_close_nei_in_back is None:
                all_close_nei_in_back = curr_close_nei
            else:
                all_close_nei_in_back = all_close_nei_in_back | curr_close_nei

    all_close_nei_in_back = all_close_nei_in_back.to(device)

    globality = create_sparse_revised(I_knn, all_close_nei_in_back, device)
    return globality

def __get_close_nei_in_back(indices, each_k_idx, cluster_labels, back_nei_idxs, k):
    # get which neighbors are close in the background set
    batch_labels = cluster_labels[each_k_idx][indices]
    top_cluster_labels = cluster_labels[each_k_idx][back_nei_idxs]
    batch_labels = repeat_1d_tensor(batch_labels, k)

    curr_close_nei = torch.eq(batch_labels, top_cluster_labels)
    return curr_close_nei

def repeat_1d_tensor(t, num_reps):
    return t.unsqueeze(1).expand(-1, num_reps)

def create_sparse_revised(I, all_close_nei_in_back, device):
    n_data, k = I.shape[0], I.shape[1]

    index = []
    similar = []
    for j in range(I.shape[0]):
        for i in range(k):
            index.append(int(j))
            similar.append(I[j][i].item())

    index = torch.masked_select(torch.LongTensor(index).to(device), all_close_nei_in_back.reshape(-1))
    similar = torch.masked_select(torch.LongTensor(similar).to(device), all_close_nei_in_back.reshape(-1))

    assert len(similar) == len(index)
    indices = torch.tensor([index.cpu().numpy().tolist(), similar.cpu().numpy().tolist()]).to(device)
    result = torch.sparse_coo_tensor(indices, torch.ones(len(index)).to(device), [n_data, n_data], dtype=torch.float).to(device)
    return result