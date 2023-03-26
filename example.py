import torch
from torch_geometric.nn.models import GCN
from torch_geometric.loader import DataLoader

from src.data import DatasetDGI
from src.datasets import Planetoid
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose

# from src.methods import DGI
from src.methods.dgi import Encoder, Method
from src.methods import DGI, DGIEncoder

# from src.methods.utils import DGIDiscriminator, DGIGCN
from src.trainer import SimpleTrainer
from src.augment import ShuffleNode

from src.evaluation import LogisticRegression

# -----------------------------------------
# Show the difference between current & old versions.
dataset_old = DatasetDGI()
dataset_old.load(path="datasets/cora_dgi")
data_old = dataset_old.to_data()
x_old = data_old.x
adj_old = data_old.adj.to_dense()

pre_transforms = Compose([NormalizeFeatures(ord=1), Edge2Adj(norm=GCNNorm(add_self_loops=1))])
dataset = Planetoid(root="pyg_data", name="cora", pre_transform=pre_transforms)
data_pyg = dataset.data
x_pyg = data_pyg.x
adj_pyg = data_pyg.adj_t.to_dense()

print("Attribute difference:", torch.sum(x_pyg - x_old))
print("Adjacency matrix difference:", torch.sum(adj_pyg - adj_old))
# ------------------------------------------

data_loader = DataLoader(dataset)

# ---------------- Fully Torch_geometric -------------------
# Neural networks
encoder = Encoder()
method = Method(encoder=xxx, in_channels=1433, hidden_channels=512)

# Trainer
trainer = SimpleTrainer(method=method, data_loader=data_loader, device="cuda:0")
trainer.train()

# Evaluation
data_pyg = data_pyg.to(method.device)
embs = method.get_embs(x=data_pyg.x, edge_index=data_pyg.edge_index, is_numpy=False)

lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=50, device="cuda")
lg(embs=embs, dataset=data_pyg)

# # ---------------- Fully Torch_geometric -------------------
# # Neural networks
# class Encoder(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels=512, num_layers=1, act=torch.nn.PReLU()):
#         super().__init__()
#         # torch_geometric build-in GCN
#         self.gcn = GCN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, act=act)
#         self.act = act
#
#     def forward(self, x, edge_index, edge_weight=None):
#         return self.act(self.gcn(x=x, edge_index=edge_index, edge_weight=edge_weight))
#
#
# encoder = Encoder(in_channels=1433)
# method = DGI(encoder=encoder, data_augment=ShuffleNode())
#
# # Trainer
# trainer = SimpleTrainer(method=method, data_loader=data_loader, device="cuda:0")
# trainer.train()
#
# # Evaluation
# data_pyg = data_pyg.to(method.device)
# embs = method.get_embs(x=data_pyg.x, edge_index=data_pyg.edge_index, is_numpy=False)
#
# lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=50, device="cuda")
# lg(embs=embs, dataset=data_pyg)

# # # ----------------- Partially Torch_geometric -----------------------
# # Neural Network
# encoder = DGIGCN(in_channels=1433, hidden_channels=512)  # the GCN implemented by DGI
# discriminator = DGIDiscriminator(in_channels=512)
# method = DGI(encoder=encoder, data_augment=ShuffleNode(), discriminator=discriminator, conv_type="spectral")
#
# # Trainer
# trainer = SimpleTrainer(method=method, data_loader=data_loader, device="cuda:0")
# trainer.train()
#
# # Evaluation
# data_pyg = data_pyg.to(method.device)
# embs = method.get_embs(x=data_pyg.x, adj=data_pyg.adj_t.to_torch_sparse_coo_tensor(), is_numpy=False)
#
# lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=50, device="cuda")
# lg(embs=embs, dataset=data_pyg)
