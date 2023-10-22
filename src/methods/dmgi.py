
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from src.methods.utils import process_dmgi
from src.nn.models.dmgi import GCN, Discriminator, Attention, AvgReadout, LogReg
import numpy as np
import time
from src.nn.models.dmgi import modeler

np.random.seed(0)

import pickle as pkl



class embedder:
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



class DMGI(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def training(self, evaluate=None):
        features = [feature.to(self.args.device) for feature in self.features]
        adj = [adj_.to(self.args.device) for adj_ in self.adj]
        model = modeler(self.args).to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef)
        cnt_wait = 0; best = 1e9
        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        for epoch in range(self.args.nb_epochs):
            xent_loss = None
            model.train()
            optimiser.zero_grad()
            idx = np.random.permutation(self.args.nb_nodes)

            shuf = [feature[:, idx, :] for feature in features]
            shuf = [shuf_ft.to(self.args.device) for shuf_ft in shuf]

            lbl_1 = torch.ones(self.args.batch_size, self.args.nb_nodes)
            lbl_2 = torch.zeros(self.args.batch_size, self.args.nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1).to(self.args.device)

            result = model(features, adj, shuf, self.args.sparse, None, None, None)
            logits = result['logits']

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
                semi_loss = xent(sup[self.idx_train], self.train_lbls)
                loss += self.args.sup_coef * semi_loss

            if loss < best:
                best = loss
                cnt_wait = 0
                torch.save(model.state_dict(), 'saved_model/best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder, self.args.metapaths))
            else:
                cnt_wait += 1

            if cnt_wait == self.args.patience:
                break

            loss.backward()
            optimiser.step()


        model.load_state_dict(torch.load('saved_model/best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder, self.args.metapaths)))

        if evaluate is not None:
            evaluate(model.H.data.detach(), self.idx_train, self.idx_val, self.idx_test, self.labels, self.args.device)