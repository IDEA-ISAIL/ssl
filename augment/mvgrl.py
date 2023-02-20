import numpy as np
import torch

from .base import DataAugmentation
from .positive import Echo
from .negative import ComputePPR, ComputeHeat

class AugPosMVGRL(DataAugmentation):
    def __init__(self, augmentors=Echo()):
        super().__init__(augmentors=augmentors)

class AugPPRMVGRL(DataAugmentation):
    def __init__(self, augmentors=ComputePPR()):
        super().__init__(augmentors=augmentors)

class AugHeatMVGRL(DataAugmentation):
    def __init__(self, augmentors=ComputeHeat()):
        super().__init__(augmentors=augmentors)        
