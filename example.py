from src.data import DatasetDGI
from src.loader import FullLoader
from src.augment.collections import augment_dgi
import torch

from src.nn.encoders import GCNDGI
from src.nn.models import ModelDGI
from src.methods import DGI
from src.datasets import Planetoid
from src.transforms import NormalizeFeatures, GCNNorm, TransformList, Edge2Adj

from torch_geometric.nn.models import GCN
from torch_geometric.loader import DataLoader

# -----------------------------------------
# Show the difference between current & old versions.
dataset_old = DatasetDGI()
dataset_old.load(path="datasets/cora_dgi")
data_old = dataset_old.to_data()
x_old = data_old.x
adj_old = data_old.adj.to_dense()

pre_transforms = TransformList([NormalizeFeatures(ord=1), Edge2Adj(norm=GCNNorm(add_self_loops=1))])
dataset = Planetoid(root="pyg_data", name="cora", pre_transform=pre_transforms)
data_pyg = dataset.data
x_pyg = data_pyg.x
adj_pyg = data_pyg.adj_t.to_dense()

print("Attribute difference:", torch.sum(x_pyg - x_old))
print("Adjacency matrix difference:", torch.sum(adj_pyg - adj_old))
# ------------------------------------------

data_loader = DataLoader(dataset)

# Neural networks
encoder = GCNDGI(dim_in=1433)
# encoder = GCN(1433, 512, num_layers=1, act="prelu")
model = ModelDGI(encoder=encoder)

# Trainer
dgi = DGI(model=model, data_loader=data_loader, data_augment=augment_dgi, save_root="./results")
dgi.train()
#
# embs = model.get_embs(x=data.x.cuda(), adj=data.adj.cuda(), is_numpy=True)
