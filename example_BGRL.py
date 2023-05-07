from src.augment import RandomMask, RandomDropEdge, RandomDropNode, AugmentSubgraph, AugmentorList, AugmentorDict, RandomMaskChannel
from src.methods import BGRL, BGRLEncoder
from src.trainer import SimpleTrainer
from torch_geometric.loader import DataLoader
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose
from src.evaluation import LogisticRegression
from src.data.data_non_contrast import Dataset
import torch, copy
import numpy as np
device="cuda:1"
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)
# WikiCS, cora, citeseer, pubmed, photo, computers, cs, and physics
data_name = "photo"
# Data
# pre_transforms = Compose([NormalizeFeatures(ord=1), Edge2Adj(norm=GCNNorm(add_self_loops=1))])
# if data_name=="cora":
#     dataset = Planetoid(root="pyg_data", name="cora", pre_transform=pre_transforms)
# elif data_name=="photo":
#     dataset = Amazon(root="pyg_data", name="Photo", pre_transform=pre_transforms)
#     dataset.data = create_masks(dataset.data, data_name)
# elif data_name=="wikics":
#     dataset = WikiCS(root="pyg_data", pre_transform=pre_transforms, is_undirected=False)
#     dataset.data = create_masks(dataset.data, data_name)
# elif data_name=="coauthorcs":
#     dataset = Coauthor(root="pyg_data", name="CS", pre_transform=pre_transforms)
#     dataset.data = create_masks(dataset.data, data_name)

dataset = Dataset(root="data", name=data_name)
if not hasattr(dataset, "adj_t"):
    data = dataset.data
    dataset.data.adj_t = torch.sparse.FloatTensor(data.edge_index, torch.ones_like(data.edge_index[0]), [data.x.shape[0], data.x.shape[0]])
print(dataset.data)
data_loader = DataLoader(dataset)
# dataset.data.x[7028] = torch.zeros((300))

# Augmentation
augment_1 = AugmentorList([RandomDropEdge(0.4), RandomMaskChannel(0.1)])
augment_2 = AugmentorList([RandomDropEdge(0.1), RandomMaskChannel(0.2)])
augment = AugmentorDict({"augment_1":augment_1, "augment_2":augment_2})

# ------------------- Method -----------------
if data_name=="cora":
    student_encoder = BGRLEncoder(in_channel=dataset.x.shape[1], hidden_channels=[2048])
elif data_name=="photo":
    student_encoder = BGRLEncoder(in_channel=dataset.x.shape[1], hidden_channels=[512, 256])
elif data_name=="WikiCS":
    student_encoder = BGRLEncoder(in_channel=dataset.x.shape[1], hidden_channels=[512, 256])
elif data_name=="coauthorcs":
    student_encoder = BGRLEncoder(in_channel=dataset.x.shape[1], hidden_channels=[256])

teacher_encoder = copy.deepcopy(student_encoder)

method = BGRL(student_encoder=student_encoder, teacher_encoder = teacher_encoder, data_augment=augment, pred_dim=512)

# ------------------ Trainer --------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader, device=device, use_ema=True, moving_average_decay=0.99, lr=1e-4, weight_decay=1e-5, dataset=dataset, n_epochs=20)
trainer.train()


# ------------------ Evaluator -------------------
method.eval()
data_pyg = dataset.data.to(method.device)
embs = method.get_embs(data_pyg, data_pyg.edge_index).detach()

lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=50, device=device)
lg(embs=embs, dataset=data_pyg)