from src.augment import RandomMask, RandomDropEdge, RandomDropNode, AugmentSubgraph, AugmentorList, AugmentorDict, NeighborSearch_AFGRL
from src.methods import AFGRLEncoder, AFGRL
from src.trainer import SimpleTrainer
from torch_geometric.loader import DataLoader
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose
from src.datasets import Planetoid, Amazon, WikiCS
from src.evaluation import LogisticRegression
import torch, copy
import torch_geometric
from src.data.utils import sparse_mx_to_torch_sparse_tensor
from src.data.data_non_contrast import Dataset
import numpy as np
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)
device="cuda:1"
# WikiCS, cora, citeseer, pubmed, photo, computers, cs, and physics
data_name = "photo"
# Data
# pre_transforms = Compose([NormalizeFeatures(ord=1), Edge2Adj(norm=GCNNorm(add_self_loops=1))])
# dataset = Planetoid(root="pyg_data", name="cora", pre_transform=pre_transforms)
# dataset = Amazon(root="pyg_data", name="Photo", pre_transform=pre_transforms)
# dataset = WikiCS(root="pyg_data", pre_transform=pre_transforms, is_undirected=False)

dataset = Dataset(root="data_", name=data_name)
if not hasattr(dataset, "adj_t"):
    data = dataset.data
    dataset.data.adj_t = torch.sparse.FloatTensor(data.edge_index, torch.ones_like(data.edge_index[0]), [data.x.shape[0], data.x.shape[0]])
print(dataset.data)
data_loader = DataLoader(dataset)
data = dataset.data
# data.x[7028] = torch.zeros((300))
adj_ori_sparse = torch.sparse.FloatTensor(data.edge_index, torch.ones_like(data.edge_index[0]), [data.x.shape[0], data.x.shape[0]]).to(device)
# Augmentation
augment = NeighborSearch_AFGRL(device=device)

# ------------------- Method -----------------
# student_encoder = AFGRLEncoder(in_channels=1433, hidden_channels=512, num_layers=1)
# student_encoder = AFGRLEncoder(in_channels=745, hidden_channels=256, num_layers=1)
# student_encoder = AFGRLEncoder(in_channels=300, hidden_channels=256, num_layers=1)
if data_name=="cora":
    student_encoder = AFGRLEncoder(in_channel=dataset.x.shape[1], hidden_channels=[2048])
elif data_name=="photo":
    student_encoder = AFGRLEncoder(in_channel=dataset.x.shape[1], hidden_channels=[512, 256])
elif data_name=="WikiCS":
    student_encoder = AFGRLEncoder(in_channel=dataset.x.shape[1], hidden_channels=[1024])
elif data_name=="coauthorcs":
    student_encoder = AFGRLEncoder(in_channel=dataset.x.shape[1], hidden_channels=[256])
teacher_encoder = copy.deepcopy(student_encoder)

method = AFGRL(student_encoder=student_encoder, teacher_encoder = teacher_encoder, data_augment=augment, adj_ori = adj_ori_sparse, topk=4)


# ------------------ Trainer --------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader, device=device, use_ema=True, moving_average_decay=0.9, lr=1e-3, weight_decay=1e-5, n_epochs=300, dataset=dataset)
trainer.train()


# ------------------ Evaluator -------------------
data_pyg = dataset.data.to(method.device)
embs = method.get_embs(data_pyg, data_pyg.edge_index).detach()

lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=50, device=device)
lg(embs=embs, dataset=data_pyg)