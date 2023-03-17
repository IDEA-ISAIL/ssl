import os
import numpy
import torch

from typing import Union, Callable, Optional, Any
from src.typing import OptAugment, Tensor


ENCODER_NAME = "encoder.ckpt"
MODEL_NAME = "model.ckpt"


class Method:
    """TODO: to be removed."""
    def __init__(
            self,
            model: torch.nn.Module,
            data_augment: OptAugment,
            emb_augment: OptAugment,
            data_loader,
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


class BaseMethod(torch.nn.Module):
    r"""Base class for self-supervised learning methods.

    Args:
        encoder (torch.nn.Module): the encoder to be trained.
        loss_function (Callable): loss function.
        data_augment (OptAugment): data augment to be used.
        emb_augment (OptAugment): embedding augment to be used.
    """
    def __init__(self,
                 encoder: torch.nn.Module,
                 loss_function: Callable,
                 data_augment: OptAugment = None,
                 emb_augment: OptAugment = None):
        super().__init__()

        self.encoder = encoder
        self.loss_function = loss_function
        self.data_augment = data_augment
        self.emb_augment = emb_augment

        self._device = None  # device: cuda or cpu
        self._param_path = None  # record the latest path used by save() or load()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def forward(self, *args, **kwargs) -> Tensor:
        r"""Perform the forward pass."""
        raise NotImplementedError

    def train_iter(self, *args, **kwargs) -> Any:
        r"""Each training iteration."""
        raise NotImplementedError

    def get_embs(self, data, is_numpy: bool = False, *args, **kwargs) -> Union[Tensor, numpy.array]:
        embs = self.encoder(data, *args, **kwargs).detach()
        if is_numpy:
            return embs.cpu().numpy()
        return embs

    def save(self, path: Optional[str] = None) -> None:
        r"""Save the parameters of the entire model to the specified path."""
        if path is None:
            path = self.__class__.__name__ + ".ckpt"
        self._param_path = path
        torch.save(self.state_dict(), self._param_path)

    def load(self, path: Optional[str] = None) -> None:
        r"""Load the parameters from the specified path."""
        if path is None:
            path = self._param_path
        if path is None:
            raise FileNotFoundError
        self._param_path = path
        state_dict = torch.load(self._param_path)
        self.model.load_state_dict(state_dict)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value
        self.to(self.device)
