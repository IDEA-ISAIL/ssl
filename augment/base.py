import logging
import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
from typing import Union, Hashable, Iterable, Optional


logger = logging.getLogger(__name__)

__all__ = [
    "BaseAugment"
]


class BaseAugment(ABC):
    @abstractmethod
    def positive(self, **kwargs):
        """
        Positive augmentation.
        """
        raise NotImplementedError

    @abstractmethod
    def negative(self, **kwargs):
        """
        Negative augmentation.
        """
        raise NotImplementedError
