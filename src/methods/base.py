import os
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

    # def __init__(self,
    #              encoder: torch.nn.Module,
    #              loss_function: Union[Callable, torch.nn.Module],
    #              data_augment: OptAugment = None,
    #              emb_augment: OptAugment = None):
    #     super().__init__()

    #     self.encoder = encoder
    #     self.loss_function = loss_function
    #     self.data_augment = data_augment
    #     self.emb_augment = emb_augment

    def __init__(self,
                 encoder: torch.nn.Module,
                 device: str="cuda",
                 save_root: str="./",
                 *args, **kwargs):
        super().__init__()

        self.encoder = encoder
        self.device = device  # device: cuda or cpu
        self.save_root = save_root  # record the latest path used by save() or load()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def forward(self, *args, **kwargs) -> Tensor:
        r"""Perform the forward pass."""
        raise NotImplementedError

    def apply_data_augment(self, *args, **kwargs) -> Any:
        r"""Apply online data augmentation."""
        raise NotImplementedError

    def apply_data_augment_offline(self, *args, **kwargs):
        r"""Apply offline data augmentation."""
        return None

    def apply_emb_augment(self, *args, **kwargs) -> Any:
        r"""Apply online embedding augmentation."""
        raise NotImplementedError

    def get_embs(self, *args, **kwargs) -> Tensor:
        embs = self.encoder(*args, **kwargs)
        return embs
    
    def get_embs(self, *args, **kwargs) -> Tensor:
        r"""Get embeddings required by the loss."""
        raise NotImplementedError

    def get_loss(self, *args, **kwargs) -> Tensor:
        r"""Get loss."""
        raise NotImplementedError

    def save(self, path: Optional[str] = None) -> None:
        r"""Save the parameters of the entire model to the specified path."""
        if path is None:
            path = self.__class__.__name__ + ".ckpt"
        path = os.path.join(self.save_root, path)
        torch.save(self.state_dict(), path)

    def load(self, path: Optional[str] = None) -> None:
        r"""Load the parameters from the specified path."""
        if path is None:
            raise FileNotFoundError
        state_dict = torch.load(path)
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

    # TODO: consider complex encoders (e.g., heterogeneous) in the future.

    def apply_data_augment_offline(self, *args, **kwargs):
        pass

    def get_view_embs(self, batch):
        r"""Get embeddings of each view.
        1. Apply data augmentation to the input batch.
        2. Use encoder to obtain the embeddings for each view.
        3. Apply embedding augmentation over the embeddings.
        """
        if self.data_augment is None:
            view2emb = {1: self.get_embs(batch.to(self._device))}
            return self.apply_emb_augment(view2emb)

        view2emb = {}
        for view, augment in self.data_augment.items():
            batch_aug = augment(batch).to(self._device)
            view2emb[view] = self.get_embs(batch_aug)
        return self.apply_emb_augment(view2emb)

    def apply_emb_augment(self, view2emb):
        # TODO: must ensure the keys of emb_augment and data_augment are the same.
        if self.emb_augment is None:
            return view2emb

        for view, augment in self.emb_augment.items():
            view2emb[view] = augment(view2emb[view]).to(self._device)
        return view2emb

    def get_data_pairs(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, batch):
        view2emb = self.get_view_embs(batch=batch)
        pos_pairs, neg_pairs = self.get_data_pairs(view2emb)
        loss = self.loss_function(pos_pairs=pos_pairs, neg_pairs=neg_pairs)
        return loss
