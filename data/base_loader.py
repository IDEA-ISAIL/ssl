from abc import ABC, abstractmethod


from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
from torch import Tensor
from torch_sparse import SparseTensor


class DataLoader(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def load(self, path):
        raise NotImplementedError
