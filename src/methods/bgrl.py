import copy
import torch
import numpy as np
from src.augment import DataAugmentation, AugNegDGI, AugPosDGI
from src.loader import Loader, FullLoader
from .base import Method
from .utils import EMA, update_moving_average

from torch_geometric.typing import *


class BGRL(Method):
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
                 moving_average_decay=0.9,
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

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr, weight_decay=weight_decay)
        self.ema_updater = EMA(moving_average_decay, n_epochs)
        # TODO: scheduler
        scheduler = lambda epoch: epoch / 1000 if epoch < 1000 \
                    else ( 1 + np.cos((epoch-1000) * np.pi / (n_epochs - 1000))) * 0.5
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = scheduler)
        # TODO: real augmentation
        self.augment_1 = lambda d: copy.deepcopy(d)
        self.augment_2 = lambda d: copy.deepcopy(d)
        self.n_epochs = n_epochs
        self.patience = patience

        self.use_cuda = use_cuda
        self.is_sparse = is_sparse

    def train(self):
        cnt_wait = 0
        best = 1e9

        data = self.data_loader.data
        n_nodes = data.n_nodes
        batch_size = self.data_loader.batch_size

        if self.use_cuda:
            self.model = self.model.cuda()

        for epoch in range(self.n_epochs):
            self.model.train()
            self.optimizer.zero_grad()

            # TODO: real data augmentation
            data_1 = self.augment_1(data)
            data_2 = self.augment_2(data)

            x_1 = data_1.x
            x_2 = data_2.x
            adj_1 = data_1.adj
            adj_2 = data_2.adj
            if self.use_cuda:
                x_1 = x_1.cuda()
                x_2 = x_2.cuda()
                adj_1 = adj_1.cuda()
                adj_2 = adj_1.cuda()

            # get loss
            loss = self.model(x_1, x_2, adj_1, adj_2)
            print(loss)
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
            self.scheduler.step()
            update_moving_average(self.ema_updater, self.model.teacher_encoder, self.model.encoder)