import torch

from src.loader import Loader, FullLoader
from .base import BaseMethod

from torch_geometric.typing import *
from scipy.sparse import coo_matrix, csr_matrix
from torch_geometric.utils import degree
from time import perf_counter as t
from tqdm import tqdm
import torch.nn.functional as F

import torch.nn as nn
from torch_geometric.typing import Tensor, Adj
from torch_geometric.nn.models import GCN

import random

def normalize_graph(A):
    eps = 2.2204e-16
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A

class SUGRL(BaseMethod):
    r"""
    TODO: add descriptions
    """
    def __init__(self,
                 encoder: torch.nn.Module,
                 data = None,
                 lr: float = 0.001,
                 weight_decay: float = 0.0,
                 n_epochs: int = 300,
                 use_cuda: bool = True,
                 is_sparse: bool = False,
                 save_root: str = "",
                 device: str = "",
                 loss_function: Optional[torch.nn.Module] = None,
                 config: dict = {}
                 ):
        super().__init__(encoder=encoder, loss_function=loss_function)
        
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
       
        self.encoder_1 = encoder[0]
        self.encoder_2 = encoder[1]
        self.data = data
        # self.model = model
        self.device = device
        self.config = config

        i = torch.LongTensor([self.data.edge_index[0].numpy(), self.data.edge_index[1].numpy()])
        v = torch.FloatTensor(torch.ones([self.data.num_edges]))
        A_sp = torch.sparse.FloatTensor(i, v, torch.Size([self.data.num_nodes, self.data.num_nodes]))
        A = A_sp.to_dense()
        I = torch.eye(A.shape[1]).to(A.device)
        A_I = A + I
        # A_nomal = normalize_graph(A)
        A_I_nomal = normalize_graph(A_I)
        self.A_I_nomal = A_I_nomal.to_sparse()
        self.lable = self.data.y
        self.nb_feature = self.data.num_features
        self.nb_classes = int(self.lable.max() - self.lable.min()) + 1
        self.nb_nodes = self.data.num_nodes
        self.data.x = torch.FloatTensor(self.data.x)
        eps = 2.2204e-16
        norm = self.data.x.norm(p=1, dim=1, keepdim=True).clamp(min=0.) + eps
        self.data.x = self.data.x.div(norm.expand_as(self.data.x))
        self.adj_1 = csr_matrix(
            (np.ones(self.data.num_edges), (self.data.edge_index[0].numpy(), self.data.edge_index[1].numpy())),
            shape=(self.data.num_nodes, self.data.num_nodes))
        self.adj_list = [self.A_I_nomal, self.adj_1]
        self.epoch = 0

    # def get_loss(self, x: Tensor, x_neg: Tensor, adj: Adj, diff: Adj, labels: Tensor):
    #     h_a = self.encoder_1(x)
    #     h_p = self.encoder_2(x, adj, is_sparse)
    #     logits, _, _ = self.model(x, x_neg, adj, diff, self.is_sparse, None, None, None)
    #     loss = self.b_xent(logits, labels)
    #     return loss

    def get_embs(self, x: Tensor, adj: Adj, is_sparse: bool = True):
        h_p = self.encoder_2(x, adj.to_torch_sparse_coo_tensor(), is_sparse)
        embs = torch.squeeze(h_p,0)
        return embs

    def forward(self,batch):
        # data, adj_list, x_list, nb_list = get_dataset(args, dataset_kwargs)
        # lable = nb_list[0]
        # nb_feature = nb_list[1]
        # nb_classes = nb_list[2]
        # nb_nodes = nb_list[3]
        
        feature_X = self.data.x.to(self.device)
        A_I_nomal = self.A_I_nomal.to(self.device)
        # tsne_lab = []
        # ylabelsx = []
        # for i in range(0, self.nb_nodes):
        #     tsne_lab.append(self.lable[i])
        #     ylabelsx.append(i)
        # tsne_lab = np.array(tsne_lab)
        # ylablesx = np.array(ylabelsx)
        # adj_1 = self.adj_1

        # adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        # test_edges, test_edges_false = mask_test_edges(adj_1, test_frac=0.1, val_frac=0.05)

        # model = self.model(nb_feature, cfg=config['cfg'],
        #                dropout=config['dropout'])
        # print(config['lr'],config['weight_decay'])
        # optimiser = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        # self.model.to(self.device)
        lable = self.lable.to(self.device)
        # train_lbls = lable[self.data.train_mask]
        # test_lbls = lable[self.data.test_mask]


        if self.config["dataset"] == 'wikics':
            train_lbls = lable[self.data.train_mask[:, self.config["NewATop"]]]  # capture
            test_lbls = lable[self.data.test_mask]
        if self.config["dataset"] in ['Cora', 'CiteSeer', 'PubMed']:
            train_lbls = lable[self.data.train_mask]
            test_lbls = lable[self.data.test_mask]
        elif self.config["dataset"] in ['photo', 'Computers', 'DBLP', 'Crocodile', 'CoraFull']:
            train_index = []
            test_index = []
            for j in range(lable.max().item() + 1):
                # num = ((lable == j) + 0).sum().item()
                index = torch.range(0, len(lable) - 1)[(lable == j).squeeze()]
                x_list0 = random.sample(list(index), int(len(index) * 0.1))
                for x in x_list0:
                    train_index.append(int(x))
            for c in range(len(lable)):
                if int(c) not in train_index:
                    test_index.append(int(c))
            train_lbls = lable[train_index].squeeze()
            test_lbls = lable[test_index]
            val_lbls = lable[train_index]

        A_degree = degree(A_I_nomal._indices()[0], self.nb_nodes, dtype=int).tolist()
        edge_index = A_I_nomal._indices()[1]
        # ===================================================#
        my_margin = self.config['margin1']
        my_margin_2 = my_margin + self.config['margin2']
        margin_loss = torch.nn.MarginRankingLoss(margin=my_margin, reduce=False).to(self.device)
        num_neg = self.config['NN']
        lbl_z = torch.tensor([0.]).to(self.device)
        deg_list_2 = []
        deg_list_2.append(0)
        for i in range(self.nb_nodes):
            deg_list_2.append(deg_list_2[-1] + A_degree[i])
        idx_p_list = []
        for j in range(1, 101):
            random_list = [deg_list_2[i] + j % A_degree[i] for i in range(self.nb_nodes)]
            idx_p = edge_index[random_list]
            idx_p_list.append(idx_p)

        idx_list = []
        for i in range(num_neg):
            idx_0 = np.random.permutation(self.nb_nodes)
            idx_list.append(idx_0)
        h_a = self.encoder_1(feature_X)
        h_p = self.encoder_2(feature_X, A_I_nomal)
        # h_a, h_p = self.model(feature_X, A_I_nomal)


        h_p_1 = (h_a[idx_p_list[self.epoch % 100]] + h_a[idx_p_list[(self.epoch + 2) % 100]] + h_a[
            idx_p_list[(self.epoch + 4) % 100]] + h_a[idx_p_list[(self.epoch + 6) % 100]] + h_a[
                    idx_p_list[(self.epoch + 8) % 100]]) / 5
        s_p = F.pairwise_distance(h_a, h_p)
        s_p_1 = F.pairwise_distance(h_a, h_p_1)
        s_p_1 = torch.unsqueeze(s_p_1, 0)
        s_n_list = []
        for h_n in idx_list:
            s_n = F.pairwise_distance(h_a, h_a[h_n])
            s_n = torch.unsqueeze(s_n, 0)
            # print(s_n.shape)
            s_n_list.append(s_n)
        margin_label = -1 * torch.ones_like(s_p)
        # print(s_n.shape)
        
        loss_mar = 0
        loss_mar_1 = 0
        mask_margin_N = 0
        for s_n in s_n_list:
            loss_mar += (margin_loss(s_p, s_n, margin_label)).mean()
            loss_mar_1 += (margin_loss(s_p_1, s_n, margin_label)).mean()
            mask_margin_N += torch.max((s_n - s_p.detach() - my_margin_2), lbl_z).sum()
        mask_margin_N = mask_margin_N / num_neg
        

        loss = loss_mar * self.config['w_loss1'] + loss_mar_1 * self.config['w_loss2'] + mask_margin_N * self.config['w_loss3']
        self.epoch += 1
        return loss
        # loss.backward()
        # optimiser.step()
        # string_1 = " loss_1: {:.3f}||loss_2: {:.3f}||loss_3: {:.3f}||".format(loss_mar.item(), loss_mar_1.item(),
                                                                                # mask_margin_N.item())
            # if args.pretrain:
            #     if os.path.exists(args.checkpoint_dir + '/' + args.dataset_name + '_weights.pth'):
            #             load_params = torch.load(args.checkpoint_dir + '/' + args.dataset_name + '_weights.pth')
            #             model_params = model.state_dict()
            #             same_parsms = {k: v for k, v in load_params.items() if k in model_params.keys()}
            #             model_params.update(same_parsms)
            #             model.load_state_dict(model_params)
            # if args.save_model:
            #     torch.save(model.state_dict(), args.checkpoint_dir + '/' + args.dataset_name + '_weights.pth')

            #evaluation

            # if epoch % config['epochs'] == 0 and epoch != 0:
            #     self.model.eval()
            #     h_a, h_p = self.model.get_embs(feature_X, A_I_nomal)
            #     embs = torch.squeeze(h_p,0)
            #     embs = embs / embs.norm(dim=1)[:, None]

              
            #     train_embs = embs[self.data.train_mask]
            #     test_embs = embs[self.data.test_mask]
                    
            #     accs = []
            #     accs_small = []
            #     xent = torch.nn.CrossEntropyLoss()
            #     for _ in range(2):
            #         log = LogReg(config['dim'], self.nb_classes)
            #         opt = torch.optim.Adam(log.parameters(), lr=config['lr2'], weight_decay=config['wd'])
            #         log.to(self.device)
            #         for _ in range(config['num1']):
            #             log.train()
            #             opt.zero_grad()
            #             logits = log(train_embs)
            #             loss = xent(logits, train_lbls)
            #             loss.backward()
            #             opt.step()
            #         logits = log(test_embs)
            #         preds = torch.argmax(logits, dim=1)
            #         acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            #         accs.append(acc * 100)
            #         ac = []
            #         for i in range(self.nb_classes):
            #             acc_small = torch.sum(preds[test_lbls == i] == test_lbls[test_lbls == i]).float() / \
            #                         test_lbls[test_lbls == i].shape[0]
            #             ac.append(acc_small * 100)
            #         accs_small = ac
            #     accs = torch.stack(accs)
            #     string_3 = ""
            #     for i in range(self.nb_classes):
            #         string_3 = string_3 + "|{:.1f}".format(accs_small[i].item())
            #     string_2 = " epoch: {},accs: {:.1f},std: {:.2f} ".format(epoch, accs.mean().item(),
            #                                                                         accs.std().item())
            #     tqdm.write(string_1 + string_2 + string_3)
            #     noe = t()
            #     print('total time', noe - start)
                # elif args.custom_key == 'link':
                #     sc_roc, sc_ap = get_roc_score(test_edges, test_edges_false, embs.cpu().detach().numpy(), adj_1)
                #     print('AUC', sc_roc, 'AP', sc_ap)
                # elif args.custom_key == 'clu':
                #     kmeans = KMeans(n_clusters=nb_classes).fit(embs[ylablesx].cpu().detach().numpy())
                #     predict_labels = kmeans.predict(embs[ylablesx].cpu().detach().numpy())

                #     cm = clustering_metrics(tsne_lab, predict_labels)
                #     acc, nmi, adjscore = cm.evaluationClusterModelFromLabel()

                #     print('Acc, nmi, adj:', acc, nmi, adjscore)

class SugrlMLP(nn.Module):
    def __init__(self, in_channels,dropout=0.2, cfg=[512, 128], batch_norm=False, out_layer=None):
        super(SugrlMLP, self).__init__()
        self.in_channels = in_channels
        self.layers = nn.ModuleList()
        self.layer_num = len(cfg)
        self.dropout = dropout
        for i in range(self.layer_num):
            if batch_norm:
                self.layers.append(nn.Linear(self.in_channels, cfg[i]))
                self.layers.append(nn.BatchNorm1d(self.out_channels, affine=False))
                self.layers.append(nn.ReLU())
            elif i != (self.layer_num-1):
                self.layers.append(nn.ReLU())
            else:
                 self.layers.append(nn.Linear(self.in_channels, cfg[i]))
            self.out_channels = cfg[i]

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        # print(self.layers)
        for i, l in enumerate(self.layers):
            # print(i)
            x = l(x)

        return x

class SugrlGCN(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 dim_out: int = 128,
                 act: torch.nn = torch.nn.PReLU(),
                 bias: bool = False):
        super(SugrlGCN, self).__init__()
        self.dim_out = dim_out
        self.fc = torch.nn.Linear(in_channels, dim_out, bias=False).to("cuda:0")
        self.act = act

        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(dim_out))
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
    def forward(self, seq, adj, is_sparse=True):
        
        seq_fts = self.fc(seq)
        if is_sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return out