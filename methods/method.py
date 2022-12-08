from typing import Tuple, List, Dict, Any
from typing import Union, Hashable, Iterable, Optional

import torch
from augment import Augment
from loader import Loader

__all__ = [
    "Method",
    "ContrastiveMethod"
]


# this is actually a trainer.
class Method:
    def __init__(
            self,
            encoder: torch.nn.Module,
            data_loader: Loader,
            data_augment: Augment,
    ) -> None:
        """
        Base class for self-supervised learning methods.

        """
        self.encoder = encoder
        self.data_iterator = data_loader
        self.data_augment = data_augment

    def get_loss(self, **kwargs):
        """
        Loss function.
        """
        raise NotImplementedError

    def train(self):
        """
        Train the encoder.
        """
        raise NotImplementedError

    def save_encoder(self, path) -> None:
        """
        Save the parameters of the encoder.
        path: path to save the parameters.
        """
        torch.save(self.encoder, path)

    def load_encoder(self, path) -> None:
        """
        Load the parameters of the encoder.
        """
        state_dict = torch.load(path)
        self.encoder.load_state_dict(state_dict)


class ContrastiveMethod(Method):
    def __init__(
            self,
            encoder: torch.nn.Module,
            data_loader: Loader,
            data_augment: Augment,
            discriminator: torch.nn.Module,
    ) -> None:
        super().__init__(
            encoder=encoder,
            data_loader=data_loader,
            data_augment=data_augment
        )

        self.discriminator = discriminator

    def train(self):
        raise NotImplementedError

    def get_loss(self, **kwargs):
        raise NotImplementedError

    def get_pos(self):
        raise NotImplementedError

    def get_neg(self):
        raise NotImplementedError

    @classmethod
    def get_label_pairs(cls, batch_size: int, n_pos: int, n_neg: int):
        """
        Get the positive and negative files
        """
        label_pos = torch.ones(batch_size, n_pos)
        label_neg = torch.zeros(batch_size, n_neg)
        labels = torch.cat((label_pos, label_neg), 1)
        return labels
