import time

import torch
from torch_geometric.loader import DataLoader

from src.methods import BaseMethod
from .base import BaseTrainer
from .utils import EarlyStopper

from typing import Union


def batch(iterable, data=None, n=1):
    l = len(iterable)
    if data == None:
        # iterable.shuffle()
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
    else:
        # iterable.shuffle()
        # data.shuffle()
        l2 = len(data)
        if n < l:
            for ndx in range(0, min(int(l2 / n), int(l / n))):
                yield (iterable.dataset[ndx * n:min(ndx * n + n, l)], data.dataset[ndx * n:min(ndx * n + n, l2)])
        else:
            for ndx in range(0, min(int(l2 / n), int(l / n))):
                yield (iterable.dataset[:], data.dataset[ndx * n:min(ndx * n + n, l2)])


class SimpleTrainer(BaseTrainer):
    r"""
    TODO: 1. Add descriptions.
          2. Do we need to support more arguments?
    """
    def __init__(self,
                 method: BaseMethod,
                 data_loader: DataLoader,
                 lr: float = 0.001,
                 weight_decay: float = 0.0,
                 n_epochs: int = 10000,
                 patience: int = 20,
                 device: Union[str, int] = "cuda:0",
                 save_root: str = "./ckpt"):
        super().__init__(method=method,
                         data_loader=data_loader,
                         save_root=save_root,
                         device=device)

        self.optimizer = torch.optim.Adam(self.method.parameters(), lr, weight_decay=weight_decay)

        self.n_epochs = n_epochs
        self.patience = patience
        self.device = device

        self.early_stopper = EarlyStopper(patience=self.patience)

    def train(self):
        self.method = self.method.to(self.device)
        self.data_loader2 = self.method.apply_data_augment_offline(self.data_loader)
        for epoch in range(self.n_epochs):
            start_time = time.time()

            for data in batch(self.data_loader, self.data_loader2):
                self.method.train()
                self.optimizer.zero_grad()

                # data = data.to(self.device)
                loss = self.method(data)

                loss.backward()
                self.optimizer.step()

            end_time = time.time()
            info = "Epoch {}: loss: {:.4f}, time: {:.4f}s".format(epoch, loss.detach().cpu().numpy(), end_time-start_time)
            print(info)

            self.early_stopper.update(loss)  # update the status
            if self.early_stopper.save:
                self.save()
            if self.early_stopper.stop:
                return
