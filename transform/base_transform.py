import logging
import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
from typing import Union, Hashable, Iterable, Optional


logger = logging.getLogger(__name__)

__all__ = [
    'BaseTransform'
]


class BaseTransform(ABC):
    @abstractmethod
    def trans_pos(self, **kwargs):
        """
        Positive transformation.
        """
        pass

    @abstractmethod
    def trans_neg(self, **kwargs):
        """
        Negative transformation.
        """
        pass


class DataTransform(BaseTransform):
    def trans_pos(self, data: Any):
        return data

    def trans_neg(self, data: Any):
        raise NotImplementedError
