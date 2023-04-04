# Major updates
## 2022.11
### 2022.11.12: Initial Push

## 2022.12
### 2022.12.12: The 1st backbone framework - <span style="color:blue">``v0.1.0``</span>

## 2023.02
### 2023.02.20: Update augmentors - <span style="color:blue">``v0.1.1``</span>
1. The `/augment` folder:
   1. Three augmentor classes: Augmentor, AugmentorList, AugmentorDict.
   2. Put single augmentors to `/augment/negative.py` & `/augment/positive.py`.
   3. Put full augmentors used in the papers to `/augment/collections.py`.
2. Split the `augment` argument in the base `Method` class into `data_augment` & `emb_augment`.
3. Add new typings.
4. Merge the MVGRL and non_contrast branches into master.

Note: In the future versions, we will:
1. Creat two sub-folders: `data` and `emb` in the `augment` folder.
2. Put all the folders into the new `ssl` folder.

## 2023.03
### 2023.03.10: Update structure 
1. Create `/src` folder to solve the problem of `typing.py`.

### 2023.03.13: Adapt Data & Datasets & Loader to torch_geometric
1. Add `/src/transforms` folder, which contains transformations/normalization for Data, 
   including feature normalization and adjacency matrix construction.
2. Update `example.py` and `/src/methods/dgi.py` based on torch_geometric.

Other updates:
1. Update the name convention of `/src/nn/models` from `xxModel` to `Model`.
2. Move `Discriminators` from `/src/nn/utils/discriminator/py` to their corresponding `/src/nn/models/xx.py`.
3. Move `Model-specific encoders` from `/src/nn/encoders/` to their corresponding `/src/nn/models/xx.py`.

### 2023.03.16: Reformulate structure of Model & Method & Trainer
1. Merge `Model` class with `Method` class, and `/src/nn/models` will be removed.
2. Add `Trainer` class to take the part of functions of the old `Method`.

### 2023.03.17: Add evaluation & the 1st overall framework test. <span style="color:blue">``v0.2.0``</span>
1. Add `BaseEvaluator` and `LogisticRegression` classes.
2. Support both `spectral` and `spatial` implementations of GCN for DGI.
3. Put the supporting functions/classes of methods to `/src/methods/utils/`, such as `DGIGCN` and `DGIDiscriminator`.
4. The 1st test for the entire framework: `Data` -> `Method` -> `Trainer` -> `Evaluator`.

Note: The performance of DGI on Cora can be reproduced by using the `spectral` convolution.
