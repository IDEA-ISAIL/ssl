
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
* DGI: [Deep Graph Infomax, ICLR'2019](https://arxiv.org/pdf/1809.10341.pdf), [(Github)](https://github.com/PetarV-/DGI)[Baoyu]
* GCC: [GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training, KDD'2020](https://arxiv.org/pdf/2006.09963.pdf), [(Github)](https://github.com/THUDM/GCC)
* GraphCL: [Graph Contrastive Learning with Augmentations](https://proceedings.nips.cc/paper/2020/file/3fe230348e9a12c13120749e3f9fa4cd-Paper.pdf), [(Github)](https://github.com/Shen-Lab/GraphCL) 
* [Graph Contrastive Learning Automated, ICML'2021](https://arxiv.org/pdf/2106.07594.pdf), [(Github)](https://github.com/Shen-Lab/GraphCL_Automated)
* MVGRL: [Contrastive Multi-View Representation Learning on Graphs](https://proceedings.mlr.press/v119/hassani20a/hassani20a.pdf), [(Github)](https://github.com/kavehhassani/mvgrl)
* GCA: [Graph Contrastive Learning with Adaptive Augmentation
](https://arxiv.org/abs/2010.14945), [(Github)](https://github.com/CRIPAC-DIG/GCA)

### non-contrastive
* BGRL: [Large-Scale Representation Learning on Graphs via Bootstrapping
](https://arxiv.org/abs/2102.06514), [(Github)](https://github.com/Namkyeong/BGRL_Pytorch)
* AFGRL

## Heterogeneous/Multiplex/Multiview
* [Unsupervised Attributed Multiplex Network Embedding, AAAI'2020](https://arxiv.org/pdf/1911.06750.pdf), [(Github)](https://github.com/pcy1302/DMGI)
* [Contrastive Multi-View Representation Learning on Graphs, ICML'2020](https://arxiv.org/pdf/2006.05582.pdf), [(Github)](https://github.com/kavehhassani/mvgrl)
* [Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning, KDD'2021](https://arxiv.org/pdf/2105.09111.pdf), [(Github)](https://github.com/liun-online/HeCo)
* [Multi-view Contrastive Graph Clustering, NeurIPS'2021](https://proceedings.neurips.cc/paper/2021/file/10c66082c124f8afe3df4886f5e516e0-Paper.pdf), [(Github)](https://github.com/Panern/MCGC)

## Temporal/Dynamic
* [Self-supervised Representation Learning on Dynamic Graphs, CIKM'2021](https://dl.acm.org/doi/pdf/10.1145/3459637.3482389), no public code
* [Mining Spatio-Temporal Relations via Self-Paced Graph Contrastive Learning, KDD'2022](https://dl.acm.org/doi/pdf/10.1145/3534678.3539422), [(Github), code is not ready on 11/29](https://github.com/RongfanLi98/SPGCL)
* [Temporality- and Frequency-aware Graph Contrastive Learning for Temporal Networks, CIKM'2022](https://dl.acm.org/doi/pdf/10.1145/3511808.3557469), [(Github)](https://github.com/ShiyinTan/TF-GCL)

## Directed
* [Directed Graph Contrastive Learning, ICML'2021](https://proceedings.neurips.cc/paper/2021/file/a3048e47310d6efaa4b1eaf55227bc92-Paper.pdf), [(Github)](https://github.com/flyingtango/DiGCL)

## Signed
* [SGCL: Contrastive Representation Learning for Signed Graphs, CIKM'2021](https://dl.acm.org/doi/pdf/10.1145/3459637.3482478), no public code

## Molecure
* InfoGraph
* JOAO
* GraphMAE

## dataset
https://github.com/dmlc/dgl/blob/master/python/dgl/data/dgl_dataset.py
https://github.com/dmlc/dgl/blob/master/python/dgl/data/gnn_benchmark.py
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/dataset.py

## data
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/data.py

## models
