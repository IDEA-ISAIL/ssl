import torch
import numpy as np
from torch_geometric.loader import DataLoader


from src.datasets import DBLP, AMiner, ACM
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose
from src.methods import HeCo, Sc_encoder, Mp_encoder, HeCoDBLPTransform
from src.trainer import SimpleTrainer
from src.evaluation import LogisticRegression
from src.utils import create_data

import pdb

# pre-tuned parameters
params = {}
params['dblp'] = {}
params['dblp']['P'] = 3
params['dblp']['sample_rate'] = [6]
params['dblp']['nei_num'] = 1
params['dblp']['target_type'] = 'author'

params['acm'] = {}
params['acm']['P'] = 2
params['acm']['sample_rate'] = [7, 1]
params['acm']['nei_num'] = 2
params['acm']['target_type'] = 'paper'
    

# -------------------- Data --------------------
# dataset = DBLP(root="DBLP_data", pre_transform=HeCoDBLPTransform())
dataset_name = 'dblp'
dataset = DBLP(root="DBLP_data", pre_transform=HeCoDBLPTransform())
data_loader = DataLoader(dataset)



# ------------------- Method -------------------
encoder1 = Mp_encoder(P=params[dataset_name]['P'], hidden_dim=64, attn_drop=0.35)
encoder2 = Sc_encoder(hidden_dim=64, sample_rate=params[dataset_name]['sample_rate'], nei_num=params[dataset_name]['nei_num'], attn_drop=0.35)

feats = data_loader.dataset._data['feats']
feats_dim_list = [i.shape[1] for i in feats]
method = HeCo(encoder1=encoder1, encoder2=encoder2, feats_dim_list = feats_dim_list)
method.cuda()


# ------------------ Trainer -------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader, device="cuda:0", n_epochs=50, lr=0.0008, patience=30)
trainer.train()


# ----------------- Evaluator ------------------
data_pyg = dataset._data[params[dataset_name]['target_type']].to(method.device)
embs = method.get_embs(data_loader.dataset._data['feats'], data_loader.dataset._data['mps']).detach()

lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=50, device="cuda")
data_pyq = create_data.create_masks(data_pyg.cpu())
lg(embs=embs, dataset=data_pyg)
