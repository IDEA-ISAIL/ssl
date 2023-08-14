import torch
import numpy as np
from torch_geometric.loader import DataLoader

from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from src.datasets import DBLP
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose
from src.methods import HeCo, Sc_encoder, Mp_encoder
from src.trainer import SimpleTrainer
from src.evaluation import LogisticRegression

import pdb

# Transform of heterogeneous dataset needs to be written dataset-specifically, since the attribute names might differ.
# ----------------- Transform ------------------
@functional_transform('heco_transform_ssl')
class HeCoTransform(BaseTransform):

    def __init__(self):
        pass

    def __call__(self, data):
        data['try'] = np.ones(2)
        return data




# -------------------- Data --------------------
dataset = DBLP(root="DBLP_data", pre_transform=HeCoTransform())
data_loader = DataLoader(dataset)

pdb.set_trace()





# ------------------- Method -------------------
encoder1 = Mp_encoder(P=3, hidden_dim=64, attn_drop=0.35)
encoder2 = Sc_encoder(hidden_dim=64, sample_rate=[6], nei_num=1, attn_drop=0.35)

method = HeCo(encoder1=encoder1, encoder2=encoder2)


# ------------------ Trainer -------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader, device="cuda:0")
trainer.train()


# ----------------- Evaluator ------------------
data_pyg = dataset.data.to(method.device)
embs = method.get_embs(data_pyg).detach()


lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=50, device="cuda")
lg(embs=embs, dataset=data_pyg)
