from .base import BaseMethod
from torch_geometric.typing import *
from src.augment import RandomMask
from typing import Optional, Callable, Union
from src.typing import AugmentType
from .utils import AvgReadout
from src.losses import NegativeMI
from copy import deepcopy

class GraphCL(BaseMethod):
    r"""
    TODO: add descriptions
    """
    def __init__(self,
                 encoder: torch.nn.Module,
                 hidden_channels: int,
                 readout: Union[Callable, torch.nn.Module] = AvgReadout(),
                 corruption: AugmentType = RandomMask(),
                 loss_function: Optional[torch.nn.Module] = None) -> None:
        loss_function = loss_function if loss_function else NegativeMI(hidden_channels)
        super().__init__(encoder=encoder, data_augment=corruption, loss_function=loss_function)

        self.readout = readout
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, batch):
        # batch2 = self.data_augment(batch).to(self._device)
        # batch2 = self.negative
        pos_batch, neg_batch = batch
        h_pos = self.encoder(pos_batch)
        h_neg = self.encoder(neg_batch)

        s = self.readout(h_pos, keepdim=True)
        s = self.sigmoid(s)
        s = s.expand_as(h_pos)

        loss = self.loss_function(x=s, y=h_pos, x_ind=s, y_ind=h_neg)
        return loss

    def apply_data_augment_offline(self, dataloader):
        # return self.data_augment(dataset).to(self._device)
        dataloader2 = deepcopy(dataloader)
        dataloader2.dataset = self.data_augment(dataloader.dataset)
        return dataloader2.to(self._device)


class GraphCLEncoder(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 512,
                 act: torch.nn = torch.nn.PReLU(),
                 bias: bool = True):
        super(GraphCLEncoder, self).__init__()
        self.dim_out = hidden_channels
        self.fc = torch.nn.Linear(in_channels, hidden_channels, bias=False)
        self.act = act

        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(hidden_channels))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self._weights_init(m)

    def _weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)

    def forward(self, batch, is_sparse=True):
        x = batch.x
        adj = batch.adj_t.to_dense()
        # edge_weight = batch.edge_weight if "edge_weight" in batch else None
        seq_fts = self.fc(x)
        if is_sparse:
            # out = torch.unsqueeze(torch.spmm(edge_index, torch.squeeze(seq_fts, 0)), 0)
            out = torch.mm(adj, torch.squeeze(seq_fts, 0))
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)

# class GraphCL(Method):
#     r"""
#     TODO: add descriptions
#     """
#     def __init__(self,
#                  model: torch.nn.Module,
#                  data_loader: Loader,
#                  data_augment: OptAugment = augment_dgi,
#                  emb_augment: OptAugment =None,
#                  lr: float = 0.001,
#                  weight_decay: float = 0.0,
#                  n_epochs: int = 10000,
#                  patience: int = 20,
#                  use_cuda: bool = True,
#                  is_sparse: bool = True,
#                  save_root: str = "",
#                  ):
#         super().__init__(model=model,
#                          data_loader=data_loader,
#                          data_augment=data_augment,
#                          emb_augment=emb_augment,
#                          save_root=save_root)
#
#
#     def get_loss(self, x: Tensor, x_neg: Tensor, adj: Adj, labels: Tensor):
#         logits = self.model(x, x_neg, adj, self.is_sparse, None, None, None)
#         loss = self.b_xent(logits, labels)
#         return loss
#
#     def get_label_pairs(self, batch_size: int, n_pos: int, n_neg: int):
#         r"""Get the positive and negative files."""
#         label_pos = torch.ones(batch_size, n_pos)
#         label_neg = torch.zeros(batch_size, n_neg)
#         labels = torch.cat((label_pos, label_neg), 1)
#         return labels
#
#     def train(self):
#         cnt_wait = 0
#         best = 1e9
#
#         data = self.data_loader.data
#
#         batch_size = self.data_loader.batch_size
#         # data augmentation
#         data_neg = self.data_augment(data)
#         data_pos = data_neg
#         adj = data_neg.adj
#         x_pos = data_pos.x
#         x_neg = data_neg.x
#         n_nodes = data_neg.n_nodes
#
#         if self.use_cuda:
#             self.model = self.model.cuda()
#             adj = adj.cuda()
#             x_pos = x_pos.cuda()
#             x_neg = x_neg.cuda()
#
#         for epoch in range(self.n_epochs):
#             self.model.train()
#             self.optimizer.zero_grad()
#
#             labels = self.get_label_pairs(batch_size=batch_size, n_pos=n_nodes, n_neg=n_nodes)
#
#             if self.use_cuda:
#                 x_neg = x_neg.cuda()
#                 labels = labels.cuda()
#
#             # get loss
#             loss = self.get_loss(x=x_pos, x_neg=x_neg, adj=adj, labels=labels)
#
#             # early stop
#             if loss < best:
#                 best = loss
#                 cnt_wait = 0
#                 self.save_model()
#                 self.save_encoder()
#             else:
#                 cnt_wait += 1
#
#             if cnt_wait == self.patience:
#                 print('Early stopping!')
#                 break
#
#             loss.backward()
#             self.optimizer.step()
