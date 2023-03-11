from src.data import DatasetDGI
from src.loader import FullLoader
from src.augment.collections import augment_dgi

from src.nn.utils import DiscriminatorDGI
from src.nn.encoders import GCNDGI
from src.nn.models import ModelDGI
from src.methods import DGI

from torch_geometric.nn.models import GCN
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader


# # data
# dataset = DatasetDGI()
# dataset.load(path="src/datasets/cora_dgi")
# data = dataset.to_data()
# data_loader = FullLoader(data)
#
# # neural networks
# encoder = GCNDGI(dim_in=1433)
# discriminator = DiscriminatorDGI(dim_h=512)
# model = ModelDGI(encoder=encoder, discriminator=discriminator)
# # model = ModelDGI(encoder=encoder)
#
# # trainer
# dgi = DGI(model=model, data_loader=data_loader, data_augment=augment_dgi, save_root="./results")
# dgi.train()
#
# embs = model.get_embs(x=data.x.cuda(), adj=data.adj.cuda(), is_numpy=True)


#### NEW ###
dataset = Planetoid(root="pyg_data", name="cora")
data_loader = DataLoader(dataset)

# neural networks
# encoder = GCNDGI(dim_in=1433)
encoder = GCN(1433, 512, num_layers=1, act="prelu")
discriminator = DiscriminatorDGI(dim_h=512)
model = ModelDGI(encoder=encoder, discriminator=discriminator)
# model = ModelDGI(encoder=encoder)

# trainer
dgi = DGI(model=model, data_loader=data_loader, data_augment=augment_dgi, save_root="./results")
dgi.train()

embs = model.get_embs(x=data.x.cuda(), adj=data.adj.cuda(), is_numpy=True)
