"""TODO: looks like this file is useless."""

import numpy as np
import torch

from .base import *
from .positive import Echo
from .negative import ComputePPR, ComputeHeat, RandomDropEdge, NeighborSearch_AFGRL, RandomMask
from .shuffle_node import ShuffleNode
from src.transforms import Compose
__all__ = [
    "augment_dgi",
    "augment_mvgrl_ppr",
    "augment_mvgrl_heat",
    "augment_bgrl_1",
    "augment_bgrl_2",
    "augment_afgrl"
]

# augment_dgi = DataShuffle(is_x=True)
augment_dgi = ShuffleNode()
augment_mvgrl_ppr = ComputePPR()
augment_mvgrl_heat = ComputeHeat()
augment_bgrl_1 = Compose([RandomMask(), RandomDropEdge()])
augment_bgrl_2 = Compose([RandomMask(), RandomDropEdge()])
augment_afgrl = NeighborSearch_AFGRL()
