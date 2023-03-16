import os
from typing import Optional, Any

import torch
from torch_geometric.loader import DataLoader
from src.methods import BaseMethod


METHOD_NAME = "method.ckpt"


class BaseTrainer(object):
    def __init__(self,
                 method: BaseMethod,
                 data_loader: DataLoader,
                 save_root: str = "./ckpt",
                 use_cuda: bool = True) -> None:
        r"""Base class for self-supervised learning methods.

        Args:
            method (torch.nn.Module): the entire method, including encoders and other components (e.g. discriminators).
            data_loader (Loader):
            save_root (str): the root to save the method/encoder.
            use_cuda (bool): whether to use cuda or not.
        """
        self.method = method
        self.data_loader = data_loader

        self.save_root = save_root
        self.use_cuda = use_cuda

        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)

    def train(self, *args, **kwargs) -> Any:
        r"""Train the encoder."""
        raise NotImplementedError

    def save(self, path: Optional[str] = None) -> None:
        r"""Save the parameters of the method."""
        if path is None:
            path = os.path.join(self.save_root, METHOD_NAME)
        self.method.save(path)

    def load(self, path: Optional[str] = None) -> None:
        r"""Load the parameters of the method."""
        if path is None:
            path = os.path.join(self.save_root, METHOD_NAME)
        self.method.load(path)
