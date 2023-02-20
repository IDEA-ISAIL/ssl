import numpy as np
import torch

from .base import *
from .positive import Echo
from .negative import DataShuffle

__all__ = [
    "augment_dgi"
]

augment_dgi = DataShuffle(is_x=True)
