# Copyright 2022 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file implements BaseMethod, which is the abstract class other
extractors will inherit from.
"""

import logging
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
from typing import Union, Hashable, Iterable, Optional

import torch
from augment import BaseAugment

logger = logging.getLogger(__name__)

__all__ = [
    "BaseMethod",
    "ContrastiveMethod"
]


# this is actually a trainer.
class BaseMethod(ABC):
    def __init__(
            self,
            encoder: torch.nn.Module,
            data_iterator: Any,
            data_augment: BaseAugment,
    ) -> None:
        """
        Base class for self-supervised learning methods.

        """
        self.encoder = encoder
        self.data_iterator = data_iterator
        self.data_augment = data_augment

    @abstractmethod
    def get_loss(self, **kwargs):
        """
        Loss function.
        """
        raise NotImplementedError

    @abstractmethod
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


class ContrastiveMethod(BaseMethod):
    def __init__(
            self,
            encoder: torch.nn.Module,
            data_iterator: Any,
            data_augment: BaseAugment,
            discriminator: torch.nn.Module,
    ) -> None:
        super().__init__(
            encoder=encoder,
            data_iterator=data_iterator,
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
