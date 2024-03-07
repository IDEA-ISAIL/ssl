
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from src.methods.utils import process_dmgi
from src.nn.models.dmgi import GCN_Encoder, Discriminator, Attention, AvgReadout, LogReg
import numpy as np
import time
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise
from tqdm import tqdm
np.random.seed(0)

import pickle as pkl



class DMGIDataset:
    def __init__(self, args):
        args.batch_size = 1
        args.sparse = True
        args.metapaths_list = args.metapaths.split(",")
        args.gpu_num_ = args.gpu_num
        if args.gpu_num_ == 'cpu':
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")

        adj, features, labels, idx_train, idx_val, idx_test = process_dmgi.load_data_dblp(args)
        features = [process_dmgi.preprocess_features(feature) for feature in features]

        args.nb_nodes = features[0].shape[0]
        args.ft_size = features[0].shape[1]
        args.nb_classes = labels.shape[1]
        args.nb_graphs = len(adj)
        args.adj = adj
        adj = [process_dmgi.normalize_adj(adj_) for adj_ in adj]
        self.adj = [process_dmgi.sparse_mx_to_torch_sparse_tensor(adj_) for adj_ in adj]

        self.features = [torch.FloatTensor(feature[np.newaxis]) for feature in features]

        self.labels = torch.FloatTensor(labels[np.newaxis]).to(args.device)
        self.idx_train = torch.LongTensor(idx_train).to(args.device)
        self.idx_val = torch.LongTensor(idx_val).to(args.device)
        self.idx_test = torch.LongTensor(idx_test).to(args.device)

        self.train_lbls = torch.argmax(self.labels[0, self.idx_train], dim=1)
        self.val_lbls = torch.argmax(self.labels[0, self.idx_val], dim=1)
        self.test_lbls = torch.argmax(self.labels[0, self.idx_test], dim=1)

        # How to aggregate
        args.readout_func = AvgReadout()

        # Summary aggregation
        args.readout_act_func = nn.Sigmoid()

        self.args = args

    def currentTime(self):
        now = time.localtime()
        s = "%04d-%02d-%02d %02d:%02d:%02d" % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        return s


class DMGI(nn.Module):
    def __init__(self, args, loss_func1, loss_func2):
        super(DMGI, self).__init__()
        self.args = args
        
        self.gcn_encoders = nn.ModuleList([GCN_Encoder(args.ft_size, args.hid_units, args.activation, args.drop_prob, args.isBias) for _ in range(args.nb_graphs)])
        self.disc = Discriminator(args.hid_units)
        
        self.H = nn.Parameter(torch.FloatTensor(1, args.nb_nodes, args.hid_units))
        self.readout_func = self.args.readout_func
        if args.isAttn:
            self.attn = nn.ModuleList([Attention(args) for _ in range(args.nheads)])

        if args.isSemi:
            self.logistic = LogReg(args.hid_units, args.nb_classes).to(args.device)

        self.init_weight()
        
        self.b_xent = loss_func1
        self.xent = loss_func2

    def init_weight(self):
        nn.init.xavier_normal_(self.H)

    def forward(self, feature, adj, shuf, sparse, msk, samp_bias1, samp_bias2, lbl=None):
        b_xent = self.b_xent
        xent = self.xent
        
        h_1_all = []; h_2_all = []; c_all = []; logits = []
        result = {}

        for i in tqdm(range(self.args.nb_graphs)):
            h_1 = self.gcn_encoders[i](feature[i], adj[i], sparse)

            # how to readout positive summary vector
            c = self.readout_func(h_1)
            c = self.args.readout_act_func(c)  # equation 9
            h_2 = self.gcn_encoders[i](shuf[i], adj[i], sparse)
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

        
        logits = result['logits']
        
        xent_loss = None

        for view_idx, logit in enumerate(logits):
            if xent_loss is None:
                xent_loss = b_xent(logit, lbl)
            else:
                xent_loss += b_xent(logit, lbl)

        loss = xent_loss

        reg_loss = result['reg_loss']
        loss += self.args.reg_coef * reg_loss

        if self.args.isSemi:
            sup = result['semi']
            semi_loss = xent(sup[self.dataset.idx_train], self.dataset.train_lbls)
            loss += self.args.sup_coef * semi_loss


        return loss



