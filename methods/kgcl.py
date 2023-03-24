import torch

import numpy as np
from loader import Loader, FullLoader
from .base import Method
from .utils import EMA, update_moving_average
from augment import kgcl
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch
import utils
from pprint import pprint
import time
from tqdm import tqdm
import multiprocessing
from sklearn.metrics import roc_auc_score
import os

from torch_geometric.typing import *
from tensorboardX import SummaryWriter
from os.path import join
import data.dataset_kgcl as dataset_kgcl



# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


def minibatch(config, *tensors, **kwargs):

    batch_size = kwargs.get('batch_size', config.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

class BPRLoss:
    def __init__(self,
                 recmodel,
                 opt,
                 config):
        self.model = recmodel
        self.opt = opt
        self.weight_decay = config.config["decay"]

    def compute(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss
        return loss

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()




CORES = multiprocessing.cpu_count() // 2

def TransR_train(recommend_model, opt, config):
    Recmodel = recommend_model
    Recmodel.train()
    kgdataset = dataset_kgcl.KGDatasetKGCL(config)
    kgloader = DataLoader(kgdataset,batch_size=4096,drop_last=True)
    trans_loss = 0.
    for data in tqdm(kgloader, total=len(kgloader), disable=True):
        heads = data[0].to(config.device)
        relations = data[1].to(config.device)
        pos_tails = data[2].to(config.device)
        neg_tails = data[3].to(config.device)
        kg_batch_loss = Recmodel.calc_kg_loss_transE(heads, relations, pos_tails, neg_tails)
        trans_loss += kg_batch_loss / len(kgloader)
        opt.zero_grad()
        kg_batch_loss.backward()
        opt.step()
    return trans_loss.cpu().item()


def train_contrast(recommend_model, augment, contrast_views, optimizer, config):
    recmodel = recommend_model
    recmodel.train()
    aver_loss = 0.

    kgv1, kgv2 = contrast_views["kgv1"], contrast_views["kgv2"]
    uiv1, uiv2 = contrast_views["uiv1"], contrast_views["uiv2"]

    l_kg = list()
    l_item = list()
    l_user = list()

    if config.kgc_enable:
        # item_num, emb_dim
        kgv1_readouts = recmodel.cal_item_embedding_from_kg(kgv1).split(2048)
        kgv2_readouts = recmodel.cal_item_embedding_from_kg(kgv2).split(2048)

        for kgv1_ro, kgv2_ro in zip(kgv1_readouts, kgv2_readouts):
            l_kg.append(augment.semi_loss(kgv1_ro, kgv2_ro).sum())

    l_contrast = torch.stack(l_kg).sum()
    optimizer.zero_grad()
    l_contrast.backward()
    optimizer.step()

    aver_loss += l_contrast.cpu().item() / len(kgv1_readouts)
    return aver_loss


def BPR_train_contrast(config, dataset, recommend_model, loss_class, augment, contrast_views, epoch, optimizer, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr = loss_class
    batch_size = config.config['bpr_batch_size']
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=12)

    total_batch = len(dataloader)
    aver_loss = 0.
    aver_loss_main = 0.
    aver_loss_ssl = 0.
    # For SGL
    uiv1, uiv2 = contrast_views["uiv1"], contrast_views["uiv2"]
    kgv1, kgv2 = contrast_views["kgv1"], contrast_views["kgv2"]
    for batch_i, train_data in tqdm(enumerate(dataloader), total=len(dataloader),disable=True):
        batch_users = train_data[0].long().to(config.device)
        batch_pos = train_data[1].long().to(config.device)
        batch_neg = train_data[2].long().to(config.device)

        # main task (batch based)
        # bpr loss for a batch of users
        l_main = bpr.compute(batch_users, batch_pos, batch_neg)
        l_ssl = list()
        items = batch_pos # [B*1]

        if config.uicontrast!="NO":
            # do SGL:
                # readout
            if config.kgc_joint:
                usersv1_ro, itemsv1_ro = Recmodel.view_computer_all(uiv1, kgv1)
                usersv2_ro, itemsv2_ro = Recmodel.view_computer_all(uiv2, kgv2)
            else:
                usersv1_ro, itemsv1_ro = Recmodel.view_computer_ui(uiv1)
                usersv2_ro, itemsv2_ro = Recmodel.view_computer_ui(uiv2)
            # from SGL source
            items_uiv1 = itemsv1_ro[items]
            items_uiv2 = itemsv2_ro[items]
            l_item = augment.info_nce_loss_overall(items_uiv1, items_uiv2, itemsv2_ro)

            users = batch_users
            users_uiv1 = usersv1_ro[users]
            users_uiv2 = usersv2_ro[users]
            l_user = augment.info_nce_loss_overall(users_uiv1, users_uiv2, usersv2_ro)
            # l_user = contrast_model.grace_loss(users_uiv1, users_uiv2)
            # L = L_main + L_user + L_item + L_kg + R^2
            l_ssl.extend([l_user*config.ssl_reg, l_item*config.ssl_reg])
        
        if l_ssl:
            l_ssl = torch.stack(l_ssl).sum()
            l_all = l_main+l_ssl
            aver_loss_ssl += l_ssl.cpu().item()
        else:
            l_all = l_main
        optimizer.zero_grad()
        l_all.backward()
        optimizer.step()

        aver_loss_main += l_main.cpu().item()
        aver_loss += l_all.cpu().item()
        if config.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', l_all, epoch * int(len(users) / config.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / (total_batch*batch_size)
    aver_loss_main = aver_loss_main / (total_batch*batch_size)
    aver_loss_ssl = aver_loss_ssl / (total_batch*batch_size)
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f} = {aver_loss_ssl:.3f}+{aver_loss_main:.3f}-{time_info}"



def BPR_train_original(config, dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr = loss_class
    
    with timer(name="Main"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(config.device)
    posItems = posItems.to(config.device)
    negItems = negItems.to(config.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // config.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(minibatch(config, users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=config.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if config.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / config.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def test_one_batch(config, X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in config.topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(config, dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = config.config['test_u_batch_size']
    testDict: dict = dataset.testDict
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(config.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(config.topks)),
               'recall': np.zeros(len(config.topks)),
               'ndcg': np.zeros(len(config.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in minibatch(config, users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(config.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(config, x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if config.tensorboard:
            w.add_scalars(f'Test/Recall@{config.topks}',
                          {str(config.topks[i]): results['recall'][i] for i in range(len(config.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{config.topks}',
                          {str(config.topks[i]): results['precision'][i] for i in range(len(config.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{config.topks}',
                          {str(config.topks[i]): results['ndcg'][i] for i in range(len(config.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results



class KGCL:
    r"""
    TODO: add descriptions
    """
    def __init__(self,
                 model: torch.nn.Module,
                 config,
                 dataset,
                 kg_dataset,
                 augment_pos,
                 lr: float = 0.001,
                 weight_decay: float = 0.0,
                 n_epochs: int = 10000,
                 moving_average_decay=0.9,
                 patience: int = 20,
                 use_cuda: bool = True,
                 is_sparse: bool = True,
                 ):
        self.Recmodel = model
        self.config = config
        self.optimizer = torch.optim.Adam(self.Recmodel.parameters(), lr=self.config.config['lr'])
        self.ema_updater = EMA(moving_average_decay, n_epochs)
        # TODO: scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1500, 2500], gamma = 0.2)
        self.n_epochs = n_epochs
        self.patience = patience

        self.use_cuda = use_cuda
        self.is_sparse = is_sparse

        self.augment = augment_pos
        self.dataset = dataset
        self.kg_dataset = kg_dataset

        self.bpr = BPRLoss(self.Recmodel, self.optimizer, self.config)

    def train(self):
        Neg_k = 1
        # init tensorboard
        if self.config.tensorboard:
            w : SummaryWriter = SummaryWriter(
                                            join(self.config.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + self.config.comment)
                                            )
        else:
            w = None
            print("not enable tensorflowboard")
        
        weight_file = self.getFileName()

        cnt_wait = 0
        best = 1e9


        least_loss = 1e5
        best_result = 0.
        stopping_step = 0

        # loss is defined in the model
        for epoch in tqdm(range(self.config.TRAIN_epochs), disable=True):
            start = time.time()
            # transR learning
            if epoch%1 == 0:
                if self.config.train_trans:
                    print("[Trans]")
                    trans_loss = TransR_train(self.Recmodel, self.optimizer, self.config)
                    print(f"trans Loss: {trans_loss:.3f}")

            
            # joint learning part
            if not self.config.pretrain_kgc:
                print("[Drop]")
                if self.config.kgc_joint:
                    contrast_views = self.augment.get_views()
                else:
                    contrast_views = self.augment.get_views("ui")
                print("[Joint Learning]")
                if self.config.kgc_joint or self.config.uicontrast!="NO":
                    output_information = BPR_train_contrast(self.config, self.dataset, self.Recmodel, self.bpr, self.augment, contrast_views, epoch, self.optimizer, neg_k=Neg_k,w=w)
                else:
                    output_information = BPR_train_original(self.confing, self.dataset, self.Recmodel, self.bpr, epoch, neg_k=Neg_k,w=w)
                

                print(f'EPOCH[{epoch+1}/{self.config.TRAIN_epochs}] {output_information}')
                if epoch<self.config.test_start_epoch:
                    if epoch %5 == 0:
                        print("[TEST]")
                        Test(self.config, self.dataset, self.Recmodel, epoch, w, self.config.config['multicore'])
                else:
                    if epoch % self.config.test_verbose == 0:
                        print("[TEST]")
                        result = Test(self.config, self.dataset, self.Recmodel, epoch, w, self.config.config['multicore'])
                        if result["recall"] > best_result:
                            stopping_step = 0
                            best_result = result["recall"]
                            print("find a better model")
                            torch.save(self.Recmodel.state_dict(), weight_file)
                        else:
                            stopping_step += 1
                            if stopping_step >= self.config.early_stop_cnt:
                                print(f"early stop triggerd at epoch {epoch}")
                                break
            
            self.scheduler.step()
    

    def getFileName(self):
        if self.config.model_name == 'mf':
            file = f"mf-{self.config.dataset}-{self.config.config['latent_dim_rec']}.pth.tar"
        elif self.config.model_name == 'lgn':
            file = f"lgn-{self.config.dataset}-{self.config.config['lightGCN_n_layers']}-{self.config.config['latent_dim_rec']}.pth.tar"
        elif self.config.model_name == 'kgcl':
            file = f"kgc-{self.config.dataset}-{self.config.config['latent_dim_rec']}.pth.tar"
        elif self.config.model_name == 'sgl':
            file = f"sgl-{self.config.dataset}-{self.config.config['latent_dim_rec']}.pth.tar"
        return os.path.join(self.config.FILE_PATH,file)