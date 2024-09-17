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
from src.config import load_yaml


from sklearn.impute import SimpleImputer
from torch_geometric.typing import SparseTensor


config = load_yaml('./configuration/gca_wikics.yml')
# config = load_yaml('./configuration/gca_coauthor.yml')
# config = load_yaml('./configuration/gca_amazon.yml')

# ---> load data
dataset_name = config.dataset.name
path = os.path.expanduser('./datasets')  
path = os.path.join(path, dataset_name)  
# print(path)

def add_adj_t(dataset):
    batch = dataset.data
    if not hasattr(batch, 'adj_t'):
        value = torch.ones_like(batch.edge_index[0], dtype=torch.float)
        batch.adj_t = SparseTensor(
            row=batch.edge_index[1],
            col=batch.edge_index[0],
            value=value,
            sparse_sizes=(batch.x.shape[0], batch.x.shape[0]),
            is_sorted=True,
            trust_data=True,
        )
    return dataset

if dataset_name == "WikiCS":
    dataset = WikiCS(root=path, transform=T.NormalizeFeatures())
    dataset = add_adj_t(dataset)
    nan_mask = torch.isnan(dataset[0].x)
    imputer = SimpleImputer()
    dataset[0].x = torch.tensor(imputer.fit_transform(dataset[0].x))
elif dataset_name == 'Amazon-Photo':
    path = os.path.join(path, dataset_name)
    dataset = Amazon(root=path, name='photo', transform=T.NormalizeFeatures())
elif dataset_name == 'Coauthor-CS':
    path = os.path.join(path, dataset_name)
    dataset = Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())
    
    
# print(dataset.data)
    
device = 'cuda:' + str(config.gpu_idx)
    

data_loader = DataLoader(dataset)
# ---> load model
encoder = GCA_Encoder(in_channels=dataset.num_features, out_channels=config.model.out_channels, activation=get_activation(config.model.activation), base_model=get_base_model('GCNConv'), k=2)
method = GRACE(encoder=encoder, loss_function=None, num_hidden=config.model.hidden_channels, num_proj_hidden=config.model.num_proj_hidden, tau=config.model.tau)

# ---> train
trainer = GCATrainer(method=method, data_loader=data_loader, drop_scheme='degree', dataset_name=dataset_name, device=device, lr=config.optim.lr, n_epochs=config.optim.max_epoch)
trainer.train()

# ---> evaluation
data = dataset.data.to(method.device)
embs = method.get_embs(data.x, data.edge_index).detach()
lg = LogisticRegression(lr=config.classifier.base_lr, weight_decay=config.classifier.weight_decay, max_iter=config.classifier.max_epoch, n_run=10, device=device)
# if dataset_name == 'WikiCS':
    # data.train_mask = torch.transpose(data.train_mask, 0, 1)
    # data.val_mask = torch.transpose(data.val_mask, 0, 1)
if dataset_name == 'Amazon-Photo' or dataset_name == 'Coauthor-CS' or dataset_name == 'WikiCS':
    data = create_data.create_masks(data.cpu())
lg(embs=embs, dataset=data)


