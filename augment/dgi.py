import numpy as np
import torch

from .base import DataAugmentation
from .positive import Echo
from .negative import Shuffle


class AugPosDGI(DataAugmentation):
    def __init__(self, augmentors=Echo()):
        super().__init__(augmentors=augmentors)


class AugNegDGI(DataAugmentation):
    def __init__(self, augmentors=Shuffle(is_x=True)):
        super().__init__(augmentors=augmentors)
