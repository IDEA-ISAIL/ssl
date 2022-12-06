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

import torch.nn

logger = logging.getLogger(__name__)

__all__ = [
    'BaseMethod'
]


# this is actually a trainer.
class BaseMethod(ABC):
    def __init__(self,
                 encoder: torch.nn.Module,
                 data_transform: Any,
                 data_iterator: Any,
                 **kwargs
                 ) -> None:
        """
        Base class for self-supervised learning methods.

        """
        self.encoder = encoder
        self.data_transform = data_transform
        self.data_iterator = data_iterator

    @abstractmethod
    def get_loss(self, **kwargs):
        """
        Loss function.
        """
        pass

    @abstractmethod
    def train(self):
        """
        Train the encoder.
        """
        pass

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
