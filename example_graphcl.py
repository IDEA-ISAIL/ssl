from src.augment import RandomMask, RandomDropEdge, RandomDropNode, AugmentSubgraph, AugmentorList
from src.methods import GraphCL, GraphCLEncoder
from src.trainer import SimpleTrainer
from torch_geometric.loader import DataLoader
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose
from src.datasets import Planetoid
from src.evaluation import LogisticRegression
import torch 

torch.manual_seed(0)
# data
pre_transforms = Compose([NormalizeFeatures(ord=1), Edge2Adj(norm=GCNNorm(add_self_loops=1))])
dataset = Planetoid(root="pyg_data", name="cora", pre_transform=pre_transforms)
data_loader = DataLoader(dataset)

# Augmentation
aug_type = 'edge'
if aug_type == 'edge':
    augment_neg = RandomDropEdge()
elif aug_type == 'mask':
    augment_neg = RandomMask()
elif aug_type == 'node':
    augment_neg = RandomDropNode()
elif aug_type == 'subgraph':
    augment_neg = AugmentSubgraph()
else:
    assert False
# augment_neg = AugmentorList([RandomDropEdge(), RandomMask()])


# ------------------- Method -----------------
encoder = GraphCLEncoder(in_channels=1433, hidden_channels=512)
method = GraphCL(encoder=encoder, corruption=augment_neg, hidden_channels=512)
method.augment_type = aug_type


# ------------------ Trainer --------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader, device="cpu")
trainer.train()


# ------------------ Evaluator -------------------
data_pyg = dataset.data.to(method.device)
embs = method.get_embs(data_pyg, data_pyg.adj_t).detach()

lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=50, device="cuda")
lg(embs=embs, dataset=data_pyg)
