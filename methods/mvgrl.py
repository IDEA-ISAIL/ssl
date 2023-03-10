import torch

from augment import augment_mvgrl_ppr
from loader import Loader, FullLoader
from .base import Method
from data.utils import sparse_mx_to_torch_sparse_tensor

from torch_geometric.typing import *
from scipy.sparse import coo_matrix


class MVGRL(Method):
    r"""
    TODO: add descriptions
    """
    def __init__(self,
                 model: torch.nn.Module,
                 data_loader: Loader,
                 data_augment: augment_mvgrl_ppr,
                 lr: float = 0.001,
                 weight_decay: float = 0.0,
                 n_epochs: int = 3000,
                 patience: int = 20,
                 sample_size: int = 2000,
                 use_cuda: bool = True,
                 is_sparse: bool = False,
                 save_root: str = "",
                 ):
        super().__init__(model=model,
                         data_augment=data_augment,
                         emb_augment=[],
                         data_loader=data_loader,
                         save_root=save_root)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
        self.b_xent = torch.nn.BCEWithLogitsLoss()

        self.n_epochs = n_epochs
        self.patience = patience
        self.sample_size = sample_size

        self.use_cuda = use_cuda
        self.is_sparse = is_sparse

    def get_loss(self, x: Tensor, x_neg: Tensor, adj: Adj, diff: Adj, labels: Tensor):
        logits, _, _ = self.model(x, x_neg, adj, diff, self.is_sparse, None, None, None)
        loss = self.b_xent(logits, labels)
        return loss

    def train(self):
        cnt_wait = 0
        best = 1e9

        data = self.data_loader.data
        batch_size = self.data_loader.batch_size
        ft_size = data.x.shape[1]

        # data augmentation
        data_neg = self.data_augment(data)

        x_pos = data.x
        adj = data.adj.to_dense()
        x_neg = data_neg.x
        diff = data_neg.adj.to_dense()

        lbl_1 = torch.ones(batch_size, self.sample_size * 2)
        lbl_2 = torch.zeros(batch_size, self.sample_size * 2)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        if self.use_cuda:
            self.model = self.model.cuda()
            lbl = lbl.cuda()

        for epoch in range(self.n_epochs):
            self.model.train()
            self.optimizer.zero_grad()

            idx = np.random.randint(0, adj.shape[-1] - self.sample_size + 1, batch_size)
            ba = torch.zeros((batch_size, self.sample_size, self.sample_size))
            bd = torch.zeros((batch_size, self.sample_size, self.sample_size))
            bf = torch.zeros((batch_size, self.sample_size, ft_size))
            for i in range(len(idx)):
                ba[i] = adj[idx[i]: idx[i] + self.sample_size, idx[i]: idx[i] + self.sample_size]
                bd[i] = diff[idx[i]: idx[i] + self.sample_size, idx[i]: idx[i] + self.sample_size]
                bf[i] = x_pos[idx[i]: idx[i] + self.sample_size]

            if self.is_sparse:
                ba = ba.to_sparse()
                bd = bd.to_sparse()

            idx = np.random.permutation(self.sample_size)
            shuf_fts = bf[:, idx, :]

            if self.use_cuda:
                x_pos = x_pos.cuda()
                x_neg = x_neg.cuda()
                adj = adj.cuda()
                diff = diff.cuda()
                bf = bf.cuda()
                ba = ba.cuda()
                bd = bd.cuda()
                shuf_fts = shuf_fts.cuda()

            # get loss
            loss = self.get_loss(x=bf, x_neg=shuf_fts, adj=ba, diff=bd, labels = lbl)

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
