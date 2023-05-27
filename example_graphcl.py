from src.augment import *
from src.methods import GraphCL, GraphCLEncoder
from src.trainer import SimpleTrainer
from torch_geometric.loader import DataLoader
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose
from src.datasets import Planetoid, Entities
from src.evaluation import LogisticRegression
import torch
from src.config import load_yaml

# load the configuration file
config = load_yaml('./configuration/graphcl.yml')
torch.manual_seed(config.torch_seed)
device = torch.device("cuda:{}".format(config.gpu_idx) if torch.cuda.is_available() and config.use_cuda else "cpu")

# data
pre_transforms = Compose([NormalizeFeatures(ord=1), Edge2Adj(norm=GCNNorm(add_self_loops=1))])
if config.dataset.root == 'pyg_data':
    dataset = Planetoid(root=config.dataset.root, name=config.dataset.name, pre_transform=pre_transforms)
else:
    raise 'please specify the correct dataset root'
# dataset = Entities(root="pyg_data", name="mutag", pre_transform=pre_transforms)
data_loader = DataLoader(dataset)

# Augmentation
aug_type = config.model.aug_type
if aug_type == 'edge':
    augment_neg = AugmentorList([RandomDropEdge()])
elif aug_type == 'mask':
    augment_neg = AugmentorList([RandomMask()])
elif aug_type == 'node':
    augment_neg = AugmentorList([RandomDropNode()])
elif aug_type == 'subgraph':
    augment_neg = AugmentorList([AugmentSubgraph()])
else:
    assert 'unrecognized augmentation method'
#
# ------------------- Method -----------------
encoder = GraphCLEncoder(in_channels=config.model.in_channels, hidden_channels=config.model.hidden_channels)
method = GraphCL(encoder=encoder, corruption=augment_neg, hidden_channels=config.model.hidden_channels)
method.augment_type = aug_type


# ------------------ Trainer --------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader, device=device)
trainer.train()


# ------------------ Evaluator -------------------
data_pyg = dataset.data.to(method.device)
embs = method.get_embs(data_pyg, data_pyg.adj_t).detach()

lg = LogisticRegression(lr=config.optim.base_lr, weight_decay=config.optim.weight_decay, max_iter=config.optim.max_epoch,
                        n_run=config.optim.run, device=device)
lg(embs=embs, dataset=data_pyg)
