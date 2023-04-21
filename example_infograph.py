import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GCN

# from src.datasets import Planetoid
# from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose
# from src.methods import DGI
from src.methods.infograph import InfoGraph, Encoder
from src.trainer import SimpleTrainer
from src.evaluation import LogisticRegression
from torch_geometric.datasets import TUDataset
import os
from torch_geometric.nn import GINConv


# -------------------- Data --------------------
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'mutag')
dataset = TUDataset(path, name='mutag').shuffle()
data_loader = DataLoader(dataset, batch_size=128)
in_channels = max(dataset.num_features, 1)

# ------------------- Method -----------------

num_layers = 2
encoder = Encoder(in_channels=in_channels, hidden_channels=32, num_layers=num_layers, GNN=GINConv)
method = InfoGraph(encoder=encoder, hidden_channels=32, num_layers=num_layers)

# ------------------ Trainer --------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader, device="cuda:0")
trainer.train()

# ------------------ Evaluator -------------------
data_pyg = dataset.data.to(method.device)
y, embs = method.get_embs(data_loader)
data_pyg.x = embs
lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=50, device="cuda")
lg(embs=embs, dataset=data_pyg)
