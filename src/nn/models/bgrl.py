import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np
import copy
from .base import BaseModel
from torch_geometric.typing import Tensor, Adj


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

    
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class Model(BaseModel):
    r"""The full model to train the encoder.

    Args:
        encoder (torch.nn.Module): the encoder to be trained.
        discriminator (torch.nn.Module): the discriminator for contrastive learning.
    """
    def __init__(self, student_encoder: torch.nn.Module, teacher_encoder: torch.nn.Module, data_augment = None):
        super().__init__(encoder=student_encoder)
        self.encoder = student_encoder
        self.teacher_encoder = teacher_encoder
        set_requires_grad(self.teacher_encoder, False)
        rep_dim = self.encoder.dim_out
        pred_hid = rep_dim*2
        self.student_predictor = nn.Sequential(nn.Linear(rep_dim, pred_hid), nn.PReLU(), nn.Linear(pred_hid, rep_dim))
        self.student_predictor.apply(init_weights)

    def forward(self, x1: Tensor, x2: Tensor, adj1: Adj, adj2: Adj, is_sparse: bool = True):
        h_1 = self.encoder(x1, adj1, is_sparse)
        h_2 = self.encoder(x2, adj2, is_sparse)

        h_1_pred = self.student_predictor(h_1)
        h_2_pred = self.student_predictor(h_2)

        with torch.no_grad():
            v_1 = self.teacher_encoder(x1, adj1, is_sparse)
            v_2 = self.teacher_encoder(x2, adj2, is_sparse)
        loss1 = loss_fn(h_1_pred, v_2.detach())
        loss2 = loss_fn(h_2_pred, v_1.detach())

        loss = loss1 + loss2
        return loss.mean()

    def get_embs(self, x: Tensor, adj: Adj, is_sparse: bool = True):
        embs = self.encoder(x=x, adj=adj, is_sparse=is_sparse)
        return embs.detach()

    def get_embs_numpy(self, x: Tensor, adj: Adj, is_sparse: bool = True):
        embs = self.get_embs(x=x, adj=adj, is_sparse=is_sparse)
        return embs.cpu().to_numpy()



