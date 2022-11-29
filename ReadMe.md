
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
[GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training](https://arxiv.org/pdf/2006.09963.pdf) [(Github)](https://github.com/THUDM/GCC)

## dataset
https://github.com/dmlc/dgl/blob/master/python/dgl/data/dgl_dataset.py
https://github.com/dmlc/dgl/blob/master/python/dgl/data/gnn_benchmark.py
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/dataset.py

## data
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/data.py

## models
