from src.data import DatasetDGI
from src.augment import ShuffleNode
import torch

from src.methods import DGI
from src.trainer import SimpleTrainer
from src.datasets import Planetoid
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose

from torch_geometric.nn.models import GCN
from torch_geometric.loader import DataLoader

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

# Neural networks
# encoder = Encoder(dim_in=1433)
encoder = GCN(1433, 512, num_layers=1, act="prelu")
method = DGI(encoder=encoder, data_augment=ShuffleNode)

# Trainer
trainer = SimpleTrainer(method=method, data_loader=data_loader)

#
<<<<<<< HEAD
#from nn.encoders import GCNDGI, GCNMVGRL
#from nn.utils import DiscriminatorDGI, DiscriminatorMVGRL
#from nn.models import ModelDGI, ModelMVGRL
#from methods import DGI, MVGRL
#
from nn.utils import DiscriminatorDGI
from nn.encoders import GCNDGI
from nn.models import ModelDGI
from methods import DGI

from src.evaluation import eval


# data
dataset = DatasetDGI()
dataset.load(path="datasets/cora_dgi")
data = dataset.to_data()
data_loader = FullLoader(data)

# neural networks
encoder = GCNDGI(dim_in=1433)
discriminator = DiscriminatorDGI(dim_h=512)
model = ModelDGI(encoder=encoder, discriminator=discriminator)
# model = ModelDGI(encoder=encoder)

# trainer
dgi = DGI(model=model, data_loader=data_loader, data_augment=augment_dgi, save_root="./results")
dgi.train()

### MVGRL ###
# dataset = DatasetMVGRL()
# dataset.load(path="./datasets/cora_dgi")
# data = dataset.to_data()
# data_loader = FullLoader(data)

# encoder_1 = GCNMVGRL(dim_in=1433)
# encoder_2 = GCNMVGRL(dim_in=1433)
# discriminator = DiscriminatorMVGRL(dim_h=512)
# model = ModelMVGRL(encoder=[encoder_1,encoder_2], discriminator=discriminator)

# augment_pos = AugPosMVGRL()
# augment_neg = AugPPRMVGRL()
# mvgrl = MVGRL(model=model, data_loader=data_loader, augment_pos=AugPosMVGRL(), augment_neg=AugPPRMVGRL(), save_root="./results")
# mvgrl.train()

#evaluator
embs = model.get_embs(x=data.x.cuda(), adj=data.adj.cuda(), is_numpy=True)
labels = dataset.labels

# node classification/ link prediction/ graph reconstruction

eval(embs, labels, evaluator = "similarity_search")



=======
# embs = model.get_embs(x=data.x.cuda(), adj=data.adj.cuda(), is_numpy=True)
>>>>>>> master
