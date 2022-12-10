import numpy as np
import torch

from augment.augment import Augment


# 假如有人想自定义augmentation怎么办？
class AugmentDGI(Augment):
    def positive(self, features: torch.Tensor):
        return features

    def negative(self, n_nodes: int, x: torch.Tensor):
        r"""Random shuffling the node attributes.

        Args:
            n_nodes (int): the number of nodes
            x (torch.Tensor): node attributes, shape: [batch_size, n_nodes, n_dims]

        Return:
            x_neg (torch.Tensor): randomly shuffled node attributes, shape [batch_size, n_nodes, n_dims]
        """
        idx = np.random.permutation(n_nodes)
        x_neg = x[:, idx, :]
        return x_neg
