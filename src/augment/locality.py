
import torch
import numpy as np
def knn_local(adj, student, teacher, top_k, device):
    n_data, d = student.shape
    similarity = torch.matmul(student, torch.transpose(teacher, 1, 0).detach())
    similarity += torch.eye(n_data, device=device) * 10

    _, I_knn = similarity.topk(k=top_k, dim=1, largest=True, sorted=True)
    # tmp = torch.LongTensor(np.arange(n_data)).unsqueeze(-1).to(device)
    knn_neighbor = create_sparse(I_knn)
    locality = knn_neighbor * adj
    return locality, I_knn

def create_sparse(I, device):
        
    similar = I.reshape(-1).tolist()
    index = np.repeat(range(I.shape[0]), I.shape[1])
    
    assert len(similar) == len(index)
    indices = torch.tensor([index, similar]).to(device)
    result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)), [I.shape[0], I.shape[0]], dtype=torch.float).to(self.device)

    return result