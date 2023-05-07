from src.augment import ComputePPR, ComputeHeat
from src.methods import GraphCL, GraphCLEncoder
from src.methods import MVGRL, MVGRLEncoder
from src.trainer import SimpleTrainer
from torch_geometric.loader import DataLoader
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose
from src.datasets import Planetoid
from src.evaluation import LogisticRegression
import torch 

torch.manual_seed(0)
# data
pre_transforms = Compose([NormalizeFeatures(ord=1), Edge2Adj(norm=GCNNorm(add_self_loops=1))])
dataset = Planetoid(root="pyg_data", name="cora", pre_transform=pre_transforms)
data_loader = DataLoader(dataset)

aug_type= 'ppr'
if aug_type == 'ppr':
    augment_neg = ComputePPR()
elif aug_type == 'heat':
    augment_neg = ComputeHeat()
else:
    assert False


# ------------------- Method -----------------
encoder = MVGRLEncoder(in_channels=1433, hidden_channels=512)
method = MVGRL(encoder=encoder, diff=augment_neg, hidden_channels=512)
method.augment_type = aug_type


# ------------------ Trainer --------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader, device="cpu", n_epochs=2000)
trainer.train()


# ------------------ Evaluator -------------------
data_pyg = dataset.data.to(method.device)
data_neg = augment_neg(data_pyg).to(method.device)
_, _, h_1, h_2, _, _ = method.get_embs(data_pyg.x, data_neg.x, data_pyg.adj_t, data_neg.adj_t, False)
embs = (h_1 + h_2).detach()

lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=50, device="cpu")
lg(embs=embs, dataset=data_pyg)

