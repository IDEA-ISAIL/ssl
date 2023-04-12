from src.augment import RandomMask, RandomDropEdge, RandomDropNode, AugmentSubgraph, AugmentorList, AugmentorDict
from src.methods import BGRL, BGRLEncoder
from src.trainer import SimpleTrainer
from torch_geometric.loader import DataLoader
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose
from src.datasets import Planetoid, Amazon, WikiCS
from src.evaluation import LogisticRegression
import torch, copy

torch.manual_seed(0)
# Data
pre_transforms = Compose([NormalizeFeatures(ord=1), Edge2Adj(norm=GCNNorm(add_self_loops=1))])
dataset = Planetoid(root="pyg_data", name="cora", pre_transform=pre_transforms)
# dataset = Amazon(root="pyg_data", name="Photo", pre_transform=pre_transforms)
# dataset = WikiCS(root="pyg_data", pre_transform=pre_transforms)
data_loader = DataLoader(dataset)

# Augmentation
augment_1 = AugmentorList([RandomDropEdge(0.3), RandomMask(0.4)])
augment_2 = AugmentorList([RandomDropEdge(0.3), RandomMask(0.4)])
augment = AugmentorDict({"augment_1":augment_1, "augment_2":augment_2})

# ------------------- Method -----------------
student_encoder = BGRLEncoder(in_channels=1433, hidden_channels=512, num_layers=1)
teacher_encoder = copy.deepcopy(student_encoder)

method = BGRL(student_encoder=student_encoder, teacher_encoder = teacher_encoder, data_augment=augment)


# ------------------ Trainer --------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader, device="cuda:0", use_ema=True, moving_average_decay=0.9, lr=1e-2)
trainer.train()


# ------------------ Evaluator -------------------
data_pyg = dataset.data.to(method.device)
embs = method.get_embs(data_pyg, data_pyg.adj_t).detach()

lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=50, device="cuda")
lg(embs=embs, dataset=data_pyg)