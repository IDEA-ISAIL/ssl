import logging
import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
from typing import Union, Hashable, Iterable, Optional


logger = logging.getLogger(__name__)

__all__ = [
    'AugmentBase'
]


class AugmentBase(ABC):
    @abstractmethod
    def augment_pos(self, **kwargs):
        """
        Positive augmentation.
        """
        raise NotImplementedError

    @abstractmethod
    def augment_neg(self, **kwargs):
        """
        Negative augmentation.
        """
        raise NotImplementedError
