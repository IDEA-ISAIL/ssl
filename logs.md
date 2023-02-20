# Major updates
## 2022.11
### 2022.11.12: Initial Push

## 2022.12
### 2022.12.12: The 1st backbone framework - v0.1.0

## 2023.02
### 2023.02.20: Major updates for augmentors - v0.1.1
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