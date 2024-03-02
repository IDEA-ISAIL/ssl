import torch.nn as nn
import torch
import torch.nn.functional as F
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import pdb
import math
from tqdm import tqdm

class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.args = args
        self.A = nn.ModuleList([nn.Linear(args.hid_units, 1) for _ in range(args.nb_graphs)])
        self.weight_init()

    def weight_init(self):
        for i in range(self.args.nb_graphs):
            nn.init.xavier_normal_(self.A[i].weight)
            self.A[i].bias.data.fill_(0.0)

    def forward(self, feat_pos, feat_neg, summary):
        feat_pos, feat_pos_attn = self.attn_feature(feat_pos)
        feat_neg, feat_neg_attn = self.attn_feature(feat_neg)
        summary, summary_attn = self.attn_summary(summary)

        return feat_pos, feat_neg, summary


    def attn_feature(self, features):
        features_attn = []
        for i in range(self.args.nb_graphs):
            features_attn.append((self.A[i](features[i].squeeze())))
        features_attn = F.softmax(torch.cat(features_attn, 1), -1)
        features = torch.cat(features,1).squeeze(0)
        features_attn_reshaped = features_attn.transpose(1, 0).contiguous().view(-1, 1)
        features = features * features_attn_reshaped.expand_as(features)
        features = features.view(self.args.nb_graphs, self.args.nb_nodes, self.args.hid_units).sum(0).unsqueeze(0)

        return features, features_attn

    def attn_summary(self, features):
        features_attn = []
        for i in range(self.args.nb_graphs):
            features_attn.append((self.A[i](features[i].squeeze())))
        features_attn = F.softmax(torch.cat(features_attn), dim=-1).unsqueeze(1)
        features = torch.cat(features, 0)
        features_attn_expanded = features_attn.expand_as(features)
        features = (features * features_attn_expanded).sum(0).unsqueeze(0)

        return features, features_attn



class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k_bilinear = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1) # c: summary vector, h_pl: positive, h_mi: negative
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k_bilinear(h_pl, c_x), 2) # sc_1 = 1 x nb_nodes
        sc_2 = torch.squeeze(self.f_k_bilinear(h_mi, c_x), 2) # sc_2 = 1 x nb_nodes

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 1)

        return logits



class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, drop_prob, isBias=False):
        super(GCN, self).__init__()

        self.fc_1 = nn.Linear(in_ft, out_ft, bias=False)

        if act == 'prelu':
            self.act = nn.PReLU()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'leakyrelu':
            self.act = nn.LeakyReLU()
        elif act == 'relu6':
            self.act = nn.ReLU6()
        elif act == 'rrelu':
            self.act = nn.RReLU()
        elif act == 'selu':
            self.act = nn.SELU()
        elif act == 'celu':
            self.act = nn.CELU()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'identity':
            self.act = nn.Identity()

        if isBias:
            self.bias_1 = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias_1.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

        self.drop_prob = drop_prob
        self.isBias = isBias
        # self.reset_parameters()

    # def reset_parameters(self):
    #     # pdb.set_trace()
    #     stdv = 1. / math.sqrt(self.fc_1.weight.data.size(0))
    #     self.fc_1.weight.data.uniform_(-stdv, stdv)
    #     if self.bias_1 is not None:
    #         self.bias_1.data.uniform_(-stdv, stdv)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq = F.dropout(seq, self.drop_prob, training=self.training)
        seq = self.fc_1(seq)
        if sparse:
            seq = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq, 0)), 0)
        else:
            seq = torch.bmm(adj, seq)

        if self.isBias:
            seq += self.bias_1

        return self.act(seq)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)



class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class DMGI(nn.Module):
    def __init__(self, args):
        super(DMGI, self).__init__()
        self.args = args
        self.gcn = nn.ModuleList([GCN(args.ft_size, args.hid_units, args.activation, args.drop_prob, args.isBias) for _ in range(args.nb_graphs)])

        self.disc = Discriminator(args.hid_units)
        self.H = nn.Parameter(torch.FloatTensor(1, args.nb_nodes, args.hid_units))
        self.readout_func = self.args.readout_func
        if args.isAttn:
            self.attn = nn.ModuleList([Attention(args) for _ in range(args.nheads)])

        if args.isSemi:
            self.logistic = LogReg(args.hid_units, args.nb_classes).to(args.device)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.H)

    def forward(self, feature, adj, shuf, sparse, msk, samp_bias1, samp_bias2):
        h_1_all = []; h_2_all = []; c_all = []; logits = []
        result = {}

        for i in tqdm(range(self.args.nb_graphs)):
            h_1 = self.gcn[i](feature[i], adj[i], sparse)

            # how to readout positive summary vector
            c = self.readout_func(h_1)
            c = self.args.readout_act_func(c)  # equation 9
            h_2 = self.gcn[i](shuf[i], adj[i], sparse)
            logit = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

            h_1_all.append(h_1)
            h_2_all.append(h_2)
            c_all.append(c)
            logits.append(logit)

        result['logits'] = logits

        # Attention or not
        if self.args.isAttn:
            h_1_all_lst = []; h_2_all_lst = []; c_all_lst = []

            for h_idx in range(self.args.nheads):
                h_1_all_, h_2_all_, c_all_ = self.attn[h_idx](h_1_all, h_2_all, c_all)
                h_1_all_lst.append(h_1_all_); h_2_all_lst.append(h_2_all_); c_all_lst.append(c_all_)

            h_1_all = torch.mean(torch.cat(h_1_all_lst, 0), 0).unsqueeze(0)
            h_2_all = torch.mean(torch.cat(h_2_all_lst, 0), 0).unsqueeze(0)

        else:
            h_1_all = torch.mean(torch.cat(h_1_all), 0).unsqueeze(0)
            h_2_all = torch.mean(torch.cat(h_2_all), 0).unsqueeze(0)


        # consensus regularizer
        pos_reg_loss = ((self.H - h_1_all) ** 2).sum()
        neg_reg_loss = ((self.H - h_2_all) ** 2).sum()
        reg_loss = pos_reg_loss - neg_reg_loss
        result['reg_loss'] = reg_loss

        # semi-supervised module
        if self.args.isSemi:
            semi = self.logistic(self.H).squeeze(0)
            result['semi'] = semi



        return result

