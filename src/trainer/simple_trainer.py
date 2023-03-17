import torch
from torch_geometric.loader import DataLoader

from src.methods import BaseMethod
from .base import BaseTrainer
from .utils import EarlyStopper


BEST_VALUE = 1e9


class SimpleTrainer(BaseTrainer):
    r"""
    TODO: add descriptions
    """

    def __init__(self,
                 method: BaseMethod,
                 data_loader: DataLoader,
                 lr: float = 0.001,
                 weight_decay: float = 0.0,
                 n_epochs: int = 10000,
                 patience: int = 20,
                 use_cuda: bool = True,
                 save_root: str = "./ckpt"):
        super().__init__(method=method,
                         data_loader=data_loader,
                         save_root=save_root,
                         use_cuda=use_cuda)

        self.optimizer = torch.optim.Adam(self.method.parameters(), lr, weight_decay=weight_decay)

        self.n_epochs = n_epochs
        self.patience = patience
        self.use_cuda = use_cuda

    def train(self):
        early_stopper = EarlyStopper(patience=self.patience, best_value=BEST_VALUE)

        if self.use_cuda:
            self.method = self.method.cuda()

        for epoch in range(self.n_epochs):
            for data in self.data_loader:
                self.method.train()
                self.optimizer.zero_grad()

                if self.use_cuda:
                    data = data.cuda()

                loss = self.method.train_iter(data)
                loss.backward()
                self.optimizer.step()

            if early_stopper.is_stop(current_value=loss):
                self.save()
                return
