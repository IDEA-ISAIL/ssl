import numpy as np
import torch

from augment.base import BaseAugment

# 假如有人想自定义augmentation怎么办？
class DGIAugment(BaseAugment):
    def positive(self, features: torch.Tensor):
        return features

    def negative(self, n_nodes: int, features: torch.Tensor):
        """
        Random shuffling the node features.
        shuf_fts: [batch_size, n_nodes, n_dims]
        """
        idx = np.random.permutation(n_nodes)
        shuf_fts = features[:, idx, :]
        return shuf_fts
