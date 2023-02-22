import torch

from augment import DataAugmentation, AugNegDGI, AugPosDGI
from loader import Loader, FullLoader
from .base import Method

from torch_geometric.typing import *


class DGI(Method):
    r"""
    TODO: add descriptions
    """
    def __init__(self,
                 model: torch.nn.Module,
                 data_loader: Loader,
                 augment_pos: DataAugmentation = AugPosDGI(),
                 augment_neg: DataAugmentation = AugNegDGI(),
                 lr: float = 0.001,
                 weight_decay: float = 0.0,
                 n_epochs: int = 10000,
                 patience: int = 20,
                 use_cuda: bool = True,
                 is_sparse: bool = True,
                 save_root: str = "",
                 ):
        super().__init__(model=model,
                         data_loader=data_loader,
                         augment_pos=augment_pos,
                         augment_neg=augment_neg,
                         save_root=save_root)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
        self.b_xent = torch.nn.BCEWithLogitsLoss()

        self.n_epochs = n_epochs
        self.patience = patience

        self.use_cuda = use_cuda
        self.is_sparse = is_sparse

    def get_loss(self, x: Tensor, x_neg: Tensor, adj: Adj, labels: Tensor):
        logits = self.model(x, x_neg, adj, self.is_sparse, None, None, None)
        loss = self.b_xent(logits, labels)
        return loss

    def train(self):
        cnt_wait = 0
        best = 1e9

        data = self.data_loader.data
        adj = data.adj
        n_nodes = data.n_nodes
        batch_size = self.data_loader.batch_size

        if self.use_cuda:
            self.model = self.model.cuda()
            adj = adj.cuda()

        for epoch in range(self.n_epochs):
            self.model.train()
            self.optimizer.zero_grad()

            # data augmentation
            data_pos = self.augment_pos(data)
            data_neg = self.augment_neg(data)

            x_pos = data_pos.x
            x_neg = data_neg.x
            if self.use_cuda:
                x_pos = x_pos.cuda()
                x_neg = x_neg.cuda()
            labels = self.get_label_pairs(batch_size=batch_size, n_pos=n_nodes, n_neg=n_nodes)

            if self.use_cuda:
                x_neg = x_neg.cuda()
                labels = labels.cuda()

            # get loss
            loss = self.get_loss(x=x_pos, x_neg=x_neg, adj=adj, labels=labels)

            # early stop
            if loss < best:
                best = loss
                cnt_wait = 0
                self.save_model()
                self.save_encoder()
            else:
                cnt_wait += 1

            if cnt_wait == self.patience:
                print('Early stopping!')
                break

            loss.backward()
            self.optimizer.step()
