[//]: # ()
[//]: # (# Plan)

[//]: # (1. Load the entire graph in the memory and use sparse computation for large graph)

[//]: # (2. Load Yaml file &#40;dictionary&#41; for hyper-parameters)

[//]: # (3. Different methods for different types of graphs)

[//]: # ()
[//]: # ()
[//]: # (# Updated file)

[//]: # (## Config.py)

[//]: # (1. model_config stores the hyper-parameters related to model, such as number of layers, hidden feature dimension, dropout rate, backbone, and etc.)

[//]: # (2. optimizer_config stores the hyper-parameters related to optimizer, such as learning rate, optimizer name &#40;e.g., adam, sgd&#41;, max epochs, patience, use_gpu, and etc.)

[//]: # (3. dataset_config stores the hyper-parameters related to dataset, such as the directory of dataset.)

[//]: # (4. ouput_config stores the hyper-parameters related to output model and output message.)

[//]: # ()
[//]: # ()
[//]: # (## main.py)

[//]: # (The main function)

[//]: # ()
[//]: # (## Dataload.py)

[//]: # (Preprocess and load the data)

[//]: # ()
[//]: # (## module.py)

[//]: # (Backbone of the graph neural network, such as GCN, GAT, GraphSAGE.)


# For Zihao

Two env: 'basket' and 'ssl'. Please execure Molecure, ReGCL, MVGRL on ssl, the others on basket.

试了下其实所有都可以在basket里跑。那就可以不用换了，其他只是当时在那个环境里调参的区别。

当时因为wikics的validation很怪或者我没看出来怎么处理，我直接把logisticregression的val mask注释了，所以这块会返回0.如果需要的话要单独处理一下wikics，或者就直接不输出validation set的结果？

amazon有人叫amazon有人叫photo，coauthor有人叫cs有人叫coauthor。

tianxin两个算法的(AFGRL,BGRL)的数据在data_里，其他有人的在pyg_data里，还有人的在datasets里……这个后面能改再改吧。


## dataset
https://github.com/dmlc/dgl/blob/master/python/dgl/data/dgl_dataset.py
https://github.com/dmlc/dgl/blob/master/python/dgl/data/gnn_benchmark.py
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/dataset.py

## data
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/data.py

# Reference Libraries
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)
- [PyGCL](https://github.com/PyGCL/PyGCL)
- [Solo-Learn](https://github.com/vturrisi/solo-learn)
- [Lightly](https://github.com/lightly-ai/lightly)
- [Faiss](https://github.com/facebookresearch/faiss)

# Paper Writing:
JMLR Paper: https://jmlr.org/papers/volume23/21-1155/21-1155.pdf
JMLR requirements: https://www.jmlr.org/mloss/mloss-info.html
Pytorch Package document: https://pytorch-geometric.readthedocs.io/en/latest/ 
https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/installation.html 
https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html

# Requirements
```
python=3.10.14
PyTorch=2.2.1+cu118
PyG=2.5.2
torch_sparse=0.6.18+pt22cu118
torch_scatter=2.1.2+pt22cu118
----------------------------
faiss-gpu=1.7.2
matplotlib=3.8.3
seaborn=0.13.2
dgl=2.1.0
```

```
conda create -n ssl python=3.10
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric==2.5.2
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install faiss-gpu==1.7.2 matplotlib==3.8.3 seaborn==0.13.2 dgl==2.1.0 pyyaml==6.0.1 pydantic==2.6.4
```