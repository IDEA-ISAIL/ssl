import torch

from src.augment.collections import augment_dgi
from src.loader import Loader
from .base import Method

from torch_geometric.typing import Tensor, Adj
from src.typing import OptAugment


class DGI(Method):
    r"""
    TODO: add descriptions
    """
    def __init__(self,
                 model: torch.nn.Module,
                 data_loader: Loader,
                 data_augment: OptAugment = augment_dgi,
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

    def get_loss(self, data, data_neg, labels: Tensor):
        logits = self.model(data, data_neg)
        # logits = self.model(x, adj)
        loss = self.b_xent(logits, labels)
        return loss

    def train(self):
        cnt_wait = 0
        best = 1e9

        if self.use_cuda:
            self.model = self.model.cuda()

        for epoch in range(self.n_epochs):
            for data in self.data_loader:
                self.model.train()
                self.optimizer.zero_grad()

                if self.use_cuda:
                    data = data.cuda()

                # data augmentation
                data_neg = self.data_augment(data)

                # labels = get_label_pairs(batch_size=len(data), n_pos=data.num_nodes, n_neg=data.num_nodes) #old
                labels = get_label_pairs(n_pos=data.num_nodes, n_neg=data.num_nodes)
                if self.use_cuda:
                    data_neg = data_neg.cuda()
                    labels = labels.cuda()

                # get loss
                loss = self.get_loss(data=data, data_neg=data_neg, labels=labels)

                # early stop
                if loss < best:
                    best = loss
                    cnt_wait = 0
                    self.save_model()
                    self.save_encoder()
                else:
                    cnt_wait += 1

                if cnt_wait == self.patience:
                    print('Early stopped!')
                    return

                loss.backward()
                self.optimizer.step()


def get_label_pairs(n_pos: int, n_neg: int):
    r"""Get the positive and negative files."""
    label_pos = torch.ones(n_pos)
    label_neg = torch.zeros(n_neg)
    labels = torch.stack((label_pos, label_neg))
    return labels
