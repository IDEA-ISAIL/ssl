# Major updates
## 2022.11
### 2022.11.12: Initial Push

## 2022.12
### 2022.12.12: The 1st backbone framework - v0.1.0

## 2023.02
### 2023.02.20: Update augmentors - v0.1.1
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

### 2023.03.13: Adapt Data \& Datasets \& Loader to torch_geometric
1. Add `/src/transforms` folder, which contains transformations/normalization for Data, 
   including feature normalization and adjacency matrix construction.
2. Update `example.py` and `/src/methods/dgi.py` based on torch_geometric.

Other updates:
1. Update the name convention of `/src/nn/models` from `xxModel` to `Model`.
2. Move `Discriminators` from `/src/nn/utils/discriminator/py` to their corresponding `/src/nn/models/xx.py`.
3. Move `Model-specific encoders` from `/src/nn/encoders/` to their corresponding `/src/nn/models/xx.py`.
