import pickle as pkl
import os
from scipy import io as sio
from sklearn.model_selection import train_test_split
from torch_sparse import SparseTensor
import torch
import numpy as np


class Graph(object):
    # graph statistics
    adj_t = None
    x = None
    y = None
    train_idx = []
    val_idx = []
    test_idx = []
    num_nodes = 0
    num_node_features = 0
    num_classes = 0


def load_graph(data_config, index=0, p=0):
    graph = Graph()
    if os.path.exists(data_config.data_dir):
        data = sio.loadmat(data_config.data_dir)
        feats = data['feats']
        labels = data['labels'].reshape(-1,)
        n_nodes = feats.shape[0]
        edge_src = data['edge_list'][:, 0]
        edge_dst = data['edge_list'][:, 1]
        train_idx = data['train_idx'][index]
        test_idx = data['test_idx'][index]
        val_idx = data['val_idx'][index]
    else:
        feats = pkl.load(open('./data/{}/'.format(data_config.name) + data_config.name + ".x.pkl", 'rb'))
        labels = pkl.load(open('./data/{}/'.format(data_config.name) + data_config.name + ".y.pkl", 'rb'))
        n_nodes = feats.shape[0]
        with open('./data/{}/'.format(data_config.name) + data_config.name + '.edgelist', 'r') as f:
            next(f)
            edge_list = np.array(list(map(lambda x: x.strip().split(' '), f.readlines())), dtype=np.int)
            edge_src = edge_list[:, 0]
            edge_dst = edge_list[:, 1]
        train, test, val = [], [], []
        for i in range(5):
            train_index, test_index, _, _ = train_test_split(range(feats.shape[0]), labels, test_size=0.87, random_state=8)
            test.append(test_index)
            train_index, val_index, _, _ = train_test_split(range(len(train_index)), labels[train_index], test_size=0.7692, random_state=8)
            train.append(train_index)
            val.append(val_index)
        data = {'feats': feats, 'labels': labels, 'edge_list': edge_list, 'train_idx': train, 'test_idx': test,
                'val_idx': val}
        sio.savemat(data_config.data_dir, data)
        train_idx = data['train_idx'][index]
        test_idx = data['test_idx'][index]
        val_idx = data['val_idx'][index]
    if p != 1:
        if not os.path.exists('./data/{}/mask_idx_{}.mat'.format(data_config.name, p*100)):
            mask_idx = np.random.binomial(1, p, feats.shape)
            mask_data = {'mask_idx': mask_idx}
            sio.savemat('./data/{}/mask_idx_{}.mat'.format(data_config.name, p*100), mask_data)
        else:
            mask_idx = sio.loadmat('./data/{}/mask_idx_{}.mat'.format(data_config.name, p*100))['mask_idx']
    else:
        mask_idx = []
    graph.train_idx = train_idx
    graph.test_idx = test_idx
    val = torch.ones(edge_src.shape, dtype=torch.float32)
    graph.adj_t = SparseTensor(row=torch.LongTensor(edge_src), col=torch.LongTensor(edge_dst), value=val,
                               sparse_sizes=(n_nodes, n_nodes))
    if len(mask_idx) != 0:
        feats[np.nonzero(mask_idx)] = 0
    graph.num_node_features = feats.shape[1]
    graph.num_classes = np.unique(labels).shape[0]
    graph.x = torch.FloatTensor(feats)
    graph.y = torch.LongTensor(labels)
    graph.val_idx = val_idx
    return graph
