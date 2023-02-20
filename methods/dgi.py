import torch

from augment.collections import augments_dgi
from loader import Loader
from .base import Method

from torch_geometric.typing import Tensor, Adj
from my_typing import OptAugment


class DGI(Method):
    r"""
    TODO: add descriptions
    """
    def __init__(self,
                 model: torch.nn.Module,
                 data_loader: Loader,
                 data_augment: OptAugment=augments_dgi,
                 lr: float = 0.001,
                 weight_decay: float = 0.0,
                 n_epochs: int = 10000,
                 patience: int = 20,
                 use_cuda: bool = True,
                 is_sparse: bool = True,
                 save_root: str = "",
                 ):
        super().__init__(model=model,
                         data_augment=data_augment,
                         emb_augment=None,
                         data_loader=data_loader,
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

    def get_label_pairs(self, batch_size: int, n_pos: int, n_neg: int):
        r"""Get the positive and negative files."""
        label_pos = torch.ones(batch_size, n_pos)
        label_neg = torch.zeros(batch_size, n_neg)
        labels = torch.cat((label_pos, label_neg), 1)
        return labels

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
            data_neg = self.data_augment(data)

            x_neg = data_neg.x
            if self.use_cuda:
                x_neg = x_neg.cuda()
            labels = self.get_label_pairs(batch_size=batch_size, n_pos=n_nodes, n_neg=n_nodes)

            if self.use_cuda:
                x_neg = x_neg.cuda()
                labels = labels.cuda()

            # get loss
            loss = self.get_loss(x=data.x, x_neg=x_neg, adj=adj, labels=labels)

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
