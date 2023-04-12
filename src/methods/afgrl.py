import torch

import numpy as np
# from augment import DataAugmentation, AugNegDGI, AugPosDGI
from src.loader import Loader, FullLoader
from .base import BaseMethod
from torch_geometric.nn.models import GCN
from torch_geometric.typing import *

import copy
from .utils import EMA, update_moving_average
import torch.nn.functional as F
from src.loader import AugmentDataLoader
import torch.nn as nn

class AFGRL(BaseMethod):
    r"""The full model to train the encoder.

    Args:
        encoder (torch.nn.Module): the encoder to be trained.
        discriminator (torch.nn.Module): the discriminator for contrastive learning.
    """
    def __init__(self, student_encoder: torch.nn.Module, teacher_encoder: torch.nn.Module, data_augment=None, adj_ori = None, topk=8):
        super().__init__(encoder=student_encoder, data_augment=data_augment, loss_function=loss_fn)
        self.encoder = student_encoder
        self.teacher_encoder = teacher_encoder
        set_requires_grad(self.teacher_encoder, False)
        rep_dim = self.encoder.hidden_channels
        pred_hid = rep_dim*2
        self.student_predictor = nn.Sequential(nn.Linear(rep_dim, pred_hid), nn.PReLU(), nn.Linear(pred_hid, rep_dim))
        self.student_predictor.apply(init_weights)
        self.data_augment = data_augment
        self.topk = topk
        self.adj_ori = adj_ori

    def forward(self, batch):
        student = self.encoder(batch, batch.edge_index)
        pred = self.student_predictor(student)

        with torch.no_grad():
            teacher = self.teacher_encoder(batch, batch.edge_index)
    
        adj_search = self.adj_ori
        student, teacher, pred = torch.squeeze(student), torch.squeeze(teacher), torch.squeeze(pred)
        ind, k = self.data_augment(adj_search, F.normalize(student, dim=-1, p=2), F.normalize(teacher, dim=-1, p=2), self.topk)

        loss1 = loss_fn(pred[ind[0]], teacher[ind[1]].detach())
        loss2 = loss_fn(pred[ind[1]], teacher[ind[0]].detach())
        loss = loss1 + loss2

        return loss.mean()

    def apply_data_augment_offline(self, dataloader):
        batch_list = []
        for i, batch in enumerate(dataloader):
            batch = batch.to(self._device)
            batch_list.append(batch)
        new_loader = AugmentDataLoader(batch_list=batch_list)
        return new_loader


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

class AFGRLEncoder(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 512,
                 act: torch.nn = torch.nn.PReLU(),
                 num_layers=1):
        super(AFGRLEncoder, self).__init__()
        self.gcn = GCN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, act=act)
        self.act = act
        self.hidden_channels = hidden_channels
        for m in self.modules():
            self._weights_init(m)

    def _weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, batch, edge_index, is_sparse=True):
        edge_weight = batch.edge_weight if "edge_weight" in batch else None
        return self.act(self.gcn(x=batch.x, edge_index=edge_index, edge_weight=edge_weight))