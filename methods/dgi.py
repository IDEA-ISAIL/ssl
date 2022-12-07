import torch
import numpy as np

from nn import DGIGCN, AvgReadout, DGIDiscriminator
from augment import BaseAugment, DGIAugment
from methods.base import ContrastiveMethod

from typing import Tuple, List, Dict, Any


class DGI_old(torch.nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI_old, self).__init__()
        self.gcn = DGIGCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = torch.nn.Sigmoid()

        self.disc = DiscriminatorDGI(n_h)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()


class DGI(ContrastiveMethod):
    def __init__(self,
                 encoder: torch.nn.Module=DGIGCN,
                 data_augment: BaseAugment=DGIAugment,
                 data_iterator: Any,
                 discriminator: torch.nn.Module=DGIDiscriminator,
                 lr: float=0.001,
                 weight_decay: float=0.0,
                 n_epochs: int=10000,
                 patience: int=20,
                 cuda: int=None,
                 sparse: bool=True
                 ):
        super().__init__(encoder=encoder,
                         data_augment=data_augment,
                         data_iterator=data_iterator,
                         discriminator=discriminator)

        self.discriminator = discriminator
        self.read = AvgReadout()
        self.sigm = torch.nn.Sigmoid()

        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr, weight_decay=weight_decay)
        self.b_xent = torch.nn.BCEWithLogitsLoss()

        self.n_epochs = n_epochs
        self.patience = patience

        self.cuda = cuda
        self.sparse = sparse

    def get_loss(self, features, shuf_fts):
        logits = self.encoder(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)
        loss = self.b_xent(logits, lbl)
        return loss

    def train(self):
        # if self.cuda:
        #     print('Using CUDA')
        #     self.encoder.cuda()
        #     features = features.cuda()
        #     if sparse:
        #         sp_adj = sp_adj.cuda()
        #     else:
        #         adj = adj.cuda()
        #     labels = labels.cuda()
        #     idx_train = idx_train.cuda()
        #     idx_val = idx_val.cuda()
        #     idx_test = idx_test.cuda()

        cnt_wait = 0
        best = 1e9
        best_t = 0

        for epoch in range(self.n_epochs):
            self.encoder.train()
            self.optimizer.zero_grad()

            # data augmentation
            feat_neg = self.data_augment.negative(n_nodes, features)

            lbl_1 = torch.ones(batch_size, n_nodes)
            lbl_2 = torch.zeros(batch_size, n_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1)

            if torch.cuda.is_available():
                shuf_fts = shuf_fts.cuda()
                lbl = lbl.cuda()


            loss = self.get_loss()

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), 'best_dgi.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == patience:
                print('Early stopping!')
                break

            loss.backward()
            self.optimizer.step()

    def _forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.encoder(seq1, adj, sparse)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.encoder(seq2, adj, sparse)

        ret = self.discriminator(c, h_1, h_2, samp_bias1, samp_bias2)
        return ret
