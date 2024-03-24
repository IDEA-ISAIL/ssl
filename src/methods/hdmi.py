import torch

import numpy as np
from src.loader import Loader, FullLoader
from .base import BaseMethod
import torch.nn.functional as F
from src.loader import AugmentDataLoader
import torch.nn as nn

from src.augment import ShuffleNode, Echo, AugmentorDict
from src.losses import NegativeMI

from typing import Optional, Callable, Union
from src.typing import AugmentType


class HDMI(BaseMethod):
    r"""The full model to train the encoder.

    Args:
        encoder (torch.nn.Module): the encoder to be trained.
        feature_size (int): the number of input features.
        hidden_channels (int): the channel of the hidden dimension.
        discriminator (torch.nn.Module): the discriminator for contrastive learning.
    """
    def __init__(self, 
                 encoder, 
                 feature_size: int, 
                 hidden_channels: int, 
                 corruption: AugmentType = ShuffleNode(), 
                 same_discriminator=True):
        super().__init__(encoder=encoder, data_augment=data_augment)

        self.disc_layers = InterDiscriminator(hidden_channels, feature_size)
        if same_discriminator:
            self.disc_fusion = self.disc_layers
        else:
            self.disc_fusion = InterDiscriminator(hidden_channels, feature_size)

    def forward(self, batch):
        h_1_list = []
        h_2_list = []
        c_list = []

        logits_e_list = []
        logits_i_list = []
        logits_j_list = []
        for i, adj in enumerate(adj_list):
            # real samples
            h_1 = torch.squeeze(self.gcn_list[i](seq1, adj, sparse))
            h_1_list.append(h_1)
            c = torch.squeeze(torch.mean(h_1, 0))   # readout
            c_list.append(c)

            # negative samples
            h_2 = torch.squeeze(self.gcn_list[i](seq2, adj, sparse))
            h_2_list.append(h_2)

            # discriminator
            logits_e, logits_i, logits_j = self.disc_layers(c, h_1, h_2, seq1, seq2)
            logits_e_list.append(logits_e)
            logits_i_list.append(logits_i)
            logits_j_list.append(logits_j)

        # fusion
        h1 = self.combine_att(h_1_list)
        h2 = self.combine_att(h_2_list)
        c = torch.mean(h1, 0)   # readout
        logits_e_fusion, logits_i_fusion, logits_j_fusion = self.disc_fusion(c, h1, h2, seq1, seq2)

        return logits_e_list, logits_i_list, logits_j_list, logits_e_fusion, logits_i_fusion, logits_j_fusion

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


class HDMI_Encoder(nn.Module):
    def __init__(self, ft_size, hid_units, n_networks):
        super(HDMI_Encoder, self).__init__()
        self.gcn_list = nn.ModuleList([GCN(ft_size, hid_units) for _ in range(n_networks)])
        self.w_list = nn.ModuleList([nn.Linear(hid_units, hid_units, bias=False) for _ in range(n_networks)])
        self.y_list = nn.ModuleList([nn.Linear(hid_units, 1) for _ in range(n_networks)])

        self.att_act1 = nn.Tanh()
        self.att_act2 = nn.Softmax(dim=-1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def combine_att(self, h_list):
        h_combine_list = []
        for i, h in enumerate(h_list):
            h = self.w_list[i](h)
            h = self.y_list[i](h)
            h_combine_list.append(h)
        score = torch.cat(h_combine_list, -1)
        score = self.att_act1(score)
        score = self.att_act2(score)
        score = torch.unsqueeze(score, -1)
        h = torch.stack(h_list, dim=1)
        h = score * h
        h = torch.sum(h, dim=1)
        return h

    def forward(self, features, adj_list, sparse=True):
        emb_list = []
        for i, adj in enumerate(adj_list):
            emb = torch.squeeze(self.gcn_list[i](features, adj, sparse))
            emb_list.append(emb)
        emb = self.combine_att(emb_list)
        emb_dict = {"final": emb, "layers": emb_list}
        return emb_dict


class InterDiscriminator(nn.Module):
    def __init__(self, n_h, ft_size):
        super().__init__()
        self.f_k_bilinear_e = nn.Bilinear(n_h, n_h, 1)
        self.f_k_bilinear_i = nn.Bilinear(ft_size, n_h, 1)
        self.f_k_bilinear_j = nn.Bilinear(n_h, n_h, 1)

        self.linear_c = nn.Linear(n_h, n_h)
        self.linear_f = nn.Linear(ft_size, n_h)
        self.linear_cf = nn.Linear(n_h*2, n_h)

        self.act = nn.Sigmoid()
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, f_pl, f_mi):
        """
        :param c: global summary vector, [dim]
        :param h_pl: positive local vectors [n_nodes, dim]
        :param h_mi: negative local vectors [n_nodes, dim]
        :param f: features/attributes [n_nodes, dim_f]
        """
        c_x = torch.unsqueeze(c, 0)
        c_x = c_x.expand_as(h_pl)
        c_x_0 = self.act(c_x)

        # extrinsic
        logits_1 = torch.squeeze(self.f_k_bilinear_e(c_x_0, h_pl))
        logits_2 = torch.squeeze(self.f_k_bilinear_e(c_x_0, h_mi))
        logits_nodes = torch.stack([logits_1, logits_2], 0)

        # intrinsic
        logits_1 = torch.squeeze(self.f_k_bilinear_i(f_pl, h_pl))
        logits_2 = torch.squeeze(self.f_k_bilinear_i(f_pl, h_mi))
        logits_locs = torch.stack([logits_1, logits_2], 0)

        # joint
        c_x = self.act(c_x)
        c_x = self.linear_c(c_x)
        f_pl = self.linear_f(f_pl)
        f_mi = self.linear_f(f_mi)
        c_x = self.act(c_x)
        f_pl = self.act(f_pl)
        f_mi = self.act(f_mi)

        cs_pl = torch.cat([c_x, f_pl], dim=-1)
        cs_mi = torch.cat([c_x, f_mi], dim=-1)

        cs_pl = self.linear_cf(cs_pl)
        cs_mi = self.linear_cf(cs_mi)
        cs_pl = self.act(cs_pl)
        cs_mi = self.act(cs_mi)

        logits_1 = torch.squeeze(self.f_k_bilinear_j(cs_pl, h_pl))
        logits_2 = torch.squeeze(self.f_k_bilinear_j(cs_mi, h_pl))
        logits_cs = torch.stack([logits_1, logits_2], 0)

        return logits_nodes, logits_locs, logits_cs


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, isBias=True):
        super(GCN, self).__init__()

        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.ReLU()

        if isBias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

        self.isBias = isBias

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq = self.fc(seq)
        if sparse:
            seq = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq, 0)), 0)
        else:
            seq = torch.bmm(adj, seq)

        if self.isBias:
            seq += self.bias

        return self.act(seq)
