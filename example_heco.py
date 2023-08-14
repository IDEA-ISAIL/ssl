import torch
from torch_geometric.loader import DataLoader

from src.datasets import Planetoid, DBLP
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose
from src.methods import HeCo, Sc_encoder, Mp_encoder
from src.trainer import SimpleTrainer
from src.evaluation import LogisticRegression

import pdb


# -------------------- Data --------------------
# TODO: only Heterogeneous transforms are supported by DBLP
# pre_transforms = Compose([NormalizeFeatures(ord=1), Edge2Adj(norm=GCNNorm(add_self_loops=1))])
dataset = DBLP(root="DBLP_data")
data_loader = DataLoader(dataset)







# ------------------- Method -------------------
encoder1 = Mp_encoder(P=3, hidden_dim=64, attn_drop=0.35)
encoder2 = Sc_encoder(hidden_dim=64, sample_rate=[6], nei_num=1, attn_drop=0.35)

method = HeCo(encoder1=encoder1, encoder2=encoder2)


# ------------------ Trainer -------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader, device="cuda:0")
trainer.train()


# ----------------- Evaluator ------------------
data_pyg = dataset.data.to(method.device)
embs = method.get_embs(data_pyg).detach()


lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=50, device="cuda")
lg(embs=embs, dataset=data_pyg)
