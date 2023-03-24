import os
import sys
import pickle as pkl
import networkx as nx
import torch_sparse
import pandas as pd
import collections
from scipy.sparse import csr_matrix
import random
from time import time
from torch.utils.data import Dataset

from .data import HomoData
#from .dataset import Dataset
from .utils import *
from os.path import join

class KGDatasetKGCL(Dataset):
    def __init__(self, config):
        self.x = None
        self.adj = None
        self.labels = None
        self.idx_train = None
        self.idx_val = None
        self.idx_test = None
        self.world = config

        self.load(join(config.DATA_PATH, config.dataset, "kg.txt"))

    def load(self, path):  # {'pubmed', 'citeseer', 'cora'}
        print("Loading data from {}".format(path))

        # need to change data type here
        kg_data = pd.read_csv(path, sep=' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        kg_dict, heads = self.generate_kg_data(kg_data=kg_data)

        self.item_net_path = join(self.world.DATA_PATH, self.world.dataset)

        self.kg_dict = kg_dict
        self.kg_data = kg_data
        self.heads = heads

    @property
    def entity_count(self):
        # start from zero
        return self.kg_data['t'].max()+2

    @property
    def relation_count(self):
        return self.kg_data['r'].max()+2

    def to_data(self):
        return HomoData(x=self.x, adj=self.adj)
        
    # KG triple list to dict
    def generate_kg_data(self, kg_data):
        kg_dict = collections.defaultdict(list)
        for row in kg_data.iterrows():
            h, r, t = row[1]
            kg_dict[h].append((r, t))
        heads = list(kg_dict.keys())
        return kg_dict, heads
    
    # item to neighbor entity matrix
    # item_num is the column num of the matrix
    def get_kg_dict(self, item_num):
        entity_num = self.world.entity_num_per_item
        i2es = dict()
        i2rs = dict()
        for item in range(item_num):
            rts = self.kg_dict.get(item, False)
            if rts:
                tails = list(map(lambda x:x[1], rts))
                relations = list(map(lambda x:x[0], rts))
                if(len(tails) > entity_num):
                    i2es[item] = torch.IntTensor(tails).to(self.world.device)[:entity_num]
                    i2rs[item] = torch.IntTensor(relations).to(self.world.device)[:entity_num]
                else:
                    # last embedding pos as padding idx
                    tails.extend([self.entity_count]*(entity_num-len(tails)))
                    relations.extend([self.relation_count]*(entity_num-len(relations)))
                    i2es[item] = torch.IntTensor(tails).to(self.world.device)
                    i2rs[item] = torch.IntTensor(relations).to(self.world.device)
            else:
                i2es[item] = torch.IntTensor([self.entity_count]*entity_num).to(self.world.device)
                i2rs[item] = torch.IntTensor([self.relation_count]*entity_num).to(self.world.device)
        return i2es, i2rs
    

    def __getitem__(self, index):
        head = self.heads[index]
        relation, pos_tail = random.choice(self.kg_dict[head])
        while True:
            neg_head = random.choice(self.heads)
            neg_tail = random.choice(self.kg_dict[neg_head])[1]
            if (relation, neg_tail) in self.kg_dict[head]:
                continue
            else:
                break
        return head, relation, pos_tail, neg_tail
    
    def __len__(self):
        return len(self.kg_dict)
    


class PurchaseDatasetKGCL(Dataset):
    def __init__(self, config, path, world):
        print(f'loading [{path}]')
        self.config = config
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.max_user_id = 0
        self.max_item_id = 0
        train_file = path + '/train.txt'
        valid_file = path + '/valid.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        validUniqueUsers, validItem, validUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0
        self.validDataSize = 0
        self.world = world

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.max_item_id = max(self.max_item_id, max(items))
                    self.max_user_id = max(self.max_user_id, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)


        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if l[1]:
                        items = [int(i) for i in l[1:]]
                        uid = int(l[0])
                        testUniqueUsers.append(uid)
                        testUser.extend([uid] * len(items))
                        testItem.extend(items)
                        self.max_item_id = max(self.max_item_id, max(items))
                        self.max_user_id = max(self.max_user_id, uid)
                        self.testDataSize += len(items)
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        if os.path.exists(valid_file):
            with open(valid_file) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        if l[1]:
                            items = [int(i) for i in l[1:]]
                            uid = int(l[0])
                            validUniqueUsers.append(uid)
                            validUser.extend([uid] * len(items))
                            validItem.extend(items)
                            self.max_item_id = max(self.max_item_id, max(items))
                            self.max_user_id = max(self.max_user_id, uid)
                            self.validDataSize += len(items)
            self.validUniqueUsers = np.array(validUniqueUsers)
            self.validUser = np.array(validUser)
            self.validItem = np.array(validItem)

        self.max_item_id += 1
        self.max_user_id += 1
        self.Graph = None
        print("interactions for training " + str(self.traindataSize))
        print(str(self.testDataSize) + " interactions for testing")
        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.max_user_id, self.max_item_id))
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.max_user_id)))
        self.__testDict = self.__build_test()
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    

    def getSparseGraph(self):
        """
        return adj matrix of user-item graph

        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(self.world.device)
                print("don't split the matrix")
        return self.Graph
    

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    
    @property
    def n_users(self):
        return self.max_user_id
    
    @property
    def m_items(self):
        return self.max_item_id
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos
    
    def __len__(self):
        return self.traindataSize
    
    def __getitem__(self, idx):
        user = self.trainUser[idx]
        pos = random.choice(self._allPos[user])
        while True:
            neg = np.random.randint(0, self.max_item_id)
            if neg in self._allPos[user]:
                continue
            else:
                break
        # return one user, one positive, one negative
        return user, pos, neg
    

