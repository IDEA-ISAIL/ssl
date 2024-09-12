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