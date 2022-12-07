import torch
import numpy as np

from nn import GCN_DGI, AvgReadout, DiscriminatorDGI
from methods.base_methods import BaseContrastiveMethod

from typing import Tuple, List, Dict, Any


class DGI_old(torch.nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI_old, self).__init__()
        self.gcn = GCN_DGI(n_in, n_h, activation)
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


class DGI(BaseContrastiveMethod):
    def __init__(self,
                 encoder: torch.nn.Module=GCN_DGI,
                 data_transform: Any,
                 data_iterator: Any,
                 discriminator: torch.nn.Module=DiscriminatorDGI,
                 lr=0.001,
                 l2_coef=0.0,
                 n_epochs=10000,
                 patience=20,
                 ):
        super().__init__(encoder=encoder,
                         data_transform=data_transform,
                         data_iterator=data_iterator,
                         discriminator=discriminator)

        self.discriminator = discriminator
        self.read = AvgReadout()
        self.sigm = torch.nn.Sigmoid()

        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr, weight_decay=l2_coef)
        self.b_xent = torch.nn.BCEWithLogitsLoss()

        self.n_epochs = n_epochs
        self.patience = patience

    def get_loss(self, features, shuf_fts):
        logits = self.encoder(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)
        loss = self.b_xent(logits, lbl)
        return loss

    def train(self):

        if torch.cuda.is_available():
            print('Using CUDA')
            model.cuda()
            features = features.cuda()
            if sparse:
                sp_adj = sp_adj.cuda()
            else:
                adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()

        cnt_wait = 0
        best = 1e9
        best_t = 0

        for epoch in range(self.n_epochs):
            self.encoder.train()
            self.optimizer.zero_grad()

            # data augmentation
            idx = np.random.permutation(nb_nodes)
            shuf_fts = features[:, idx, :]

            lbl_1 = torch.ones(batch_size, nb_nodes)
            lbl_2 = torch.zeros(batch_size, nb_nodes)
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
