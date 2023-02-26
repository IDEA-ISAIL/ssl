import numpy as np
import torch

from .base import *
from .positive import Echo
from .negative import DataShuffle, ComputePPR, ComputeHeat

__all__ = [
    "augment_dgi",
    "augment_mvgrl_ppr",
    "augment_mvgrl_heat",
]

augment_dgi = DataShuffle(is_x=True)
augment_mvgrl_ppr = ComputePPR()
augment_mvgrl_heat = ComputeHeat()