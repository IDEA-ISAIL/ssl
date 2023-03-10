import os
from typing import Optional
from src.typing import OptAugment

import torch
from src.loader import Loader

__all__ = [
    "Method",
]

ENCODER_NAME = "encoder.ckpt"
MODEL_NAME = "model.ckpt"


# this is actually a trainer.
class Method:
    def __init__(
            self,
            model: torch.nn.Module,
            data_augment: OptAugment,
            emb_augment: OptAugment,
            data_loader: Loader,
            save_root: str = "",
            use_cuda: bool = True,
            *args,
            **kwargs
    ) -> None:
        r"""Base class for self-supervised learning methods.

        Args:
            model (torch.nn.Module): the entire model, including encoders and other components (e.g. discriminators).
            data_loader (Loader):
            save_root (str): the root to save the model/encoder.
            use_cuda (bool): whether to use cuda or not.
            *args:
            **kwargs:
        """
        self.model = model  # entire model to train, including encoders and other necessary modules
        self.data_augment = data_augment
        self.emb_augment = emb_augment
        self.data_loader = data_loader
        self.save_root = save_root
        self.use_cuda = use_cuda

        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)

    def get_loss(self, *args, **kwargs):
        r"""Loss function."""
        raise NotImplementedError

    def train(self, *args, **kwargs):
        r"""Train the encoder."""
        raise NotImplementedError

    def save_encoder(self, path: Optional[str] = None) -> None:
        r"""Save the parameters of the encoder.

        Args:
            path (Optional[str]): path to save the parameters.
        """
        if path is None:
            path = os.path.join(self.save_root, ENCODER_NAME)
        torch.save(self.model.encoder, path)

    def save_model(self, path: Optional[str] = None) -> None:
        r"""Save the parameters of the entire model.

        Args:
            path (Optional[str]): path to save the parameters.
        """
        if path is None:
            path = os.path.join(self.save_root, MODEL_NAME)
        torch.save(self.model.encoder, path)

    def load_encoder(self, path: str) -> None:
        r"""Load the parameters of the encoder."""
        state_dict = torch.load(path)
        self.model.encoder.load_state_dict(state_dict)

    def load_model(self, path: str) -> None:
        r"""Load the parameters of the entire model."""
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
