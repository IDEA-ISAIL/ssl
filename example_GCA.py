import os
import torch

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

from src.methods.gca import GCA_Encoder, GRACE
from src.datasets import WikiCS, Amazon, Coauthor
from src.augment.gca_augments import get_activation, get_base_model
from src.trainer.gca_trainer import GCATrainer

from src.evaluation import LogisticRegression
from src.utils import create_data

# ---> load data
# dataset_name = 'WikiCS'
dataset_name = 'Amazon-Photo'
# dataset_name = 'Coauthor-CS'
path = os.path.expanduser('~/datasets')  # C:\Users\dongqif2\datasets
path = os.path.join(path, dataset_name)  # C:\Users\dongqif2\datasets\WikiCS
dataset = WikiCS(root=path, transform=T.NormalizeFeatures())
data_loader = DataLoader(dataset)
if dataset_name == 'Amazon-Photo':
    path = os.path.join(path, dataset_name)
    dataset = Amazon(root=path, name='photo', transform=T.NormalizeFeatures())
    data_loader = DataLoader(dataset)
if dataset_name == 'Coauthor-CS':
    path = os.path.join(path, dataset_name)
    dataset = Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())
    data_loader = DataLoader(dataset)


# ---> load model
encoder = GCA_Encoder(in_channels=dataset.num_features, out_channels=256, activation=get_activation('prelu'), base_model=get_base_model('GCNConv'), k=2)
method = GRACE(encoder=encoder, loss_function=None, num_hidden=256, num_proj_hidden=32, tau=0.4)

# ---> train
trainer = GCATrainer(method=method, data_loader=data_loader, drop_scheme='degree', dataset_name='WikiCS', device="cuda:1")
trainer.train()

# ---> evaluation
data = dataset.data.to(method.device)
embs = method.get_embs(data.x, data.edge_index).detach()
lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=50, device="cuda:1")
if dataset_name == 'WikiCS':
    data.train_mask = torch.transpose(data.train_mask, 0, 1)
    data.val_mask = torch.transpose(data.val_mask, 0, 1)
if dataset_name == 'Amazon-Photo' or dataset_name == 'Coauthor-CS':
    data = create_data.create_masks(data.cpu())
lg(embs=embs, dataset=data)