
# Plan
1. Load the entire graph in the memory and use sparse computation for large graph
2. Load Yaml file (dictionary) for hyper-parameters
3. Different methods for different types of graphs


# Updated file
## Config.py
1. model_config stores the hyper-parameters related to model, such as number of layers, hidden feature dimension, dropout rate, backbone, and etc.
2. optimizer_config stores the hyper-parameters related to optimizer, such as learning rate, optimizer name (e.g., adam, sgd), max epochs, patience, use_gpu, and etc.
3. dataset_config stores the hyper-parameters related to dataset, such as the directory of dataset.
4. ouput_config stores the hyper-parameters related to output model and output message.


## main.py
The main function

## Dataload.py
Preprocess and load the data

## module.py
Backbone of the graph neural network, such as GCN, GAT, GraphSAGE.

# Reference files
## General
[Deep Graph Infomax, ICLR'2019](https://arxiv.org/pdf/1809.10341.pdf), [(Github)](https://github.com/PetarV-/DGI)
[GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training, KDD'2020](https://arxiv.org/pdf/2006.09963.pdf), [(Github)](https://github.com/THUDM/GCC)

## Heterogeneous
[Unsupervised Attributed Multiplex Network Embedding, AAAI'2020](https://arxiv.org/pdf/1911.06750.pdf), [(Github)](https://github.com/pcy1302/DMGI)
[Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning, KDD'2021](https://arxiv.org/pdf/2105.09111.pdf), [(Github)](https://github.com/liun-online/HeCo)

## Temporal
[Mining Spatio-Temporal Relations via Self-Paced Graph Contrastive Learning, KDD'2022](https://dl.acm.org/doi/pdf/10.1145/3534678.3539422), [(Github), code is not ready on 11/29](https://github.com/RongfanLi98/SPGCL)

## dataset
https://github.com/dmlc/dgl/blob/master/python/dgl/data/dgl_dataset.py
https://github.com/dmlc/dgl/blob/master/python/dgl/data/gnn_benchmark.py
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/dataset.py

## data
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/data.py

## models
