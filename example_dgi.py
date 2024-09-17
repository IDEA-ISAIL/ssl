import torch
from torch_geometric.loader import DataLoader
# from torch_geometric.nn.models import GCN

from src.datasets import Planetoid, Amazon, WikiCS,Coauthor
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose
from src.methods import DGI, DGIEncoder
from src.trainer import SimpleTrainer
from src.augment import ShuffleNode
import torch_geometric.transforms as T
from src.evaluation import LogisticRegression, SVCRegression, RandomForestClassifier, TSNEVisulization, SimSearch, K_Means
from src.utils.add_adj import add_adj_t
from src.data.data_non_contrast import Dataset


import yaml

seed = 0
device = 'cuda:0'

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# -------------------- Data --------------------
pre_transforms = Compose([NormalizeFeatures(ord=1), Edge2Adj(norm=GCNNorm(add_self_loops=1))])
# dataset = Planetoid(root="pyg_data", name="cora", pre_transform=pre_transforms)
# data_loader = DataLoader(dataset)

# load the configuration file
# config = yaml.safe_load(open('./configuration/dgi_wikics.yml', 'r', encoding='utf-8').read())
# config = yaml.safe_load(open("./configuration/dgi_amazon.yml", 'r', encoding='utf-8').read())
config = yaml.safe_load(open("./configuration/dgi_coauthor.yml", 'r', encoding='utf-8').read())
print(config)
data_name = config['dataset']

# if data_name=="cora":
#     dataset = Planetoid(root="pyg_data", name="cora", pre_transform=pre_transforms)
# if data_name=="photo": 
#     dataset = Amazon(root="pyg_data", name="photo", pre_transform=pre_transforms) 
# elif data_name=="coauthor": 
#     dataset = Coauthor(root="pyg_data", name='cs', transform=pre_transforms)
# elif data_name=="wikics": 
#     dataset = WikiCS(root="pyg_data", transform=T.NormalizeFeatures())
#     dataset = add_adj_t(dataset)
#     nan_mask = torch.isnan(dataset[0].x)
#     imputer = SimpleImputer()
#     dataset[0].x = torch.tensor(imputer.fit_transform(dataset[0].x))

root = config['root']
dataset = Dataset(root=root, name=data_name)
if data_name in ['Amazon', 'WikiCS', 'coauthor']:
    dataset.data = create_masks(dataset.data, config.dataset.name)
dataset = add_adj_t(dataset)
data_loader = DataLoader(dataset)

# print(dataset.data)


# ------------------- Method -----------------
# class Encoder(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels=512, num_layers=1, act=torch.nn.PReLU()):
#         super().__init__()
#         self.gcn = GCN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, act=act)
#         self.act = act

#     def forward(self, batch):
#         edge_weight = batch.edge_weight if "edge_weight" in batch else None
#         return self.act(self.gcn(x=batch.x, edge_index=batch.edge_index, edge_weight=edge_weight))

encoder = DGIEncoder(in_channels=config['in_channels'], hidden_channels=512)
method = DGI(encoder=encoder, hidden_channels=512)

# ------------------ Trainer --------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader, device=device,
                        n_epochs=config['epochs'],
                        lr=config['lr'], 
                        patience=config['patience'])
trainer.train()

# ------------------ Evaluator -------------------
data_pyg = dataset.data.to(method.device)
embs = encoder(data_pyg).detach()

lg = LogisticRegression(lr=0.01,
                        weight_decay=0,
                        max_iter=1500,
                        n_run=10,
                        device=device)
lg(embs=embs, dataset=data_pyg)

# plot_embedding2D(embs=embs, dataset=data_pyg)
