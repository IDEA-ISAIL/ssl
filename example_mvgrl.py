from src.augment import ComputePPR, ComputeHeat
from src.methods import MVGRL, MVGRLEncoder
from src.trainer import SimpleTrainer
from torch_geometric.loader import DataLoader
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose
from src.datasets import Planetoid, Entities, Amazon, WikiCS, Coauthor
from src.evaluation import LogisticRegression
import torch
from src.config import load_yaml
from src.utils.create_data import create_masks
from src.utils.add_adj import add_adj_t

# load the configuration file
# config = load_yaml('./configuration/mvgrl_amazon.yml')
# config = load_yaml('./configuration/mvgrl_coauthor.yml')
# config = load_yaml('./configuration/mvgrl_wikics.yml')
config = load_yaml('./configuration/mvgrl_cora.yml')
torch.manual_seed(config.torch_seed)
# device = torch.device("cuda:{}".format(config.gpu_idx) if torch.cuda.is_available() and config.use_cuda else "cpu")
device = 'cpu'

# data
if config.dataset.name == 'cora':
    pre_transforms = Compose([NormalizeFeatures(ord=1), Edge2Adj(norm=GCNNorm(add_self_loops=1))])
    # dataset = Planetoid(root=config.dataset.root, name=config.dataset.name, pre_transform=pre_transforms)
    dataset = Planetoid(root='pyg_data', name='cora', pre_transform=pre_transforms)
elif config.dataset.name == 'Amazon':
    pre_transforms = NormalizeFeatures(ord=1)
    dataset = Amazon(root='pyg_data', name='Photo', pre_transform=pre_transforms)
elif config.dataset.name == 'WikiCS':
    pre_transforms = NormalizeFeatures(ord=1)
    dataset = WikiCS(root='pyg_data', pre_transform=pre_transforms)
elif config.dataset.name == 'coauthor':
    pre_transforms = NormalizeFeatures(ord=1)
    dataset = Coauthor(root='pyg_data', name='CS', pre_transform=pre_transforms)
else:
    raise 'please specify the correct dataset root'
data_loader = DataLoader(dataset, batch_size=config.model.batch_size)

# Augmentation
aug_type = config.model.aug_type
if aug_type == 'ppr':
    augment_neg = ComputePPR(alpha = config.model.alpha)
elif aug_type == 'heat':
    augment_neg = ComputeHeat(t = config.model.t)
else:
    assert False


# ------------------- Method -----------------
encoder = MVGRLEncoder(in_channels=config.model.in_channels, hidden_channels=config.model.hidden_channels)
method = MVGRL(encoder=encoder, diff=augment_neg, hidden_channels=config.model.hidden_channels)
method.augment_type = aug_type


# ------------------ Trainer --------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader, device=device, n_epochs=config.optim.max_epoch, patience=config.optim.patience)
trainer.train()


# ------------------ Evaluator -------------------
data_pyg = dataset.data.to(method.device)
data_neg = augment_neg(data_pyg).to(method.device)
_, _, h_1, h_2, _, _ = method.get_embs(data_pyg.x, data_neg.x, data_pyg.adj_t, data_neg.adj_t, False)
embs = (h_1 + h_2).detach()

lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=50, device="cpu")
lg(embs=embs, dataset=data_pyg)

