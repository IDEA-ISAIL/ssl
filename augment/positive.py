import copy
import torch

from .base import Augmentor
from data import Data, HomoData


class Echo(Augmentor):
    def __init__(self):
        super().__init__()

    def apply(self, data: Data):
        return data
