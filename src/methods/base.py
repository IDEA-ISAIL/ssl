import numpy
import torch

from typing import Union, Callable, Optional, Any
from src.typing import OptAugment, Tensor

ENCODER_NAME = "encoder.ckpt"
MODEL_NAME = "model.ckpt"


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
                 loss_function: Union[Callable, torch.nn.Module],
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

    def apply_data_augment(self, *args, **kwargs) -> Any:
        r"""Apply online data augmentation."""
        raise NotImplementedError

    def apply_data_augment_offline(self, *args, **kwargs):
        r"""Apply offline data augmentation."""
        raise NotImplementedError

    def apply_emb_augment(self, *args, **kwargs) -> Any:
        r"""Apply online embedding augmentation."""
        raise NotImplementedError

    def get_embs(self, is_numpy: bool = False, *args, **kwargs) -> Union[Tensor, numpy.array]:
        embs = self.encoder(*args, **kwargs).detach()
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


class ContrastiveMethod(BaseMethod):
    def __init__(self,
                 encoder: torch.nn.Module,
                 loss_function: Union[Callable, torch.nn.Module],
                 data_augment: OptAugment = None,
                 emb_augment: OptAugment = None):
        super().__init__(encoder=encoder,
                         loss_function=loss_function,
                         data_augment=data_augment,
                         emb_augment=emb_augment)

    def apply_data_augment_offline(self, *args, **kwargs):
        pass

    def train_iter(self, batch):
        aug2data = self.apply_data_augment(batch)
        tmp = self.forward(aug2data)
        pos_pairs, neg_pairs = self.get_data_pairs(*tmp)
        loss = self.loss_function(pos_pairs=pos_pairs, neg_pairs=neg_pairs)
        return loss

    def apply_data_augment(self, batch):
        if self.data_augment is None:
            return batch

        aug2data = {}
        for key, value in self.data_augment.items():
            aug2data[key] = value(batch).to(self._device)
        return aug2data

    def apply_emb_augment(self, *args, **kwargs):
        pass

    def get_data_pairs(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError
