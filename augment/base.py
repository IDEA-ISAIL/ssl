from data import Data

from abc import ABC
from typing import List


__all__ = [
    "DataAugmentation"
]


class Augmentation:
    r"""Base class for augmentation."""
    def __init__(self):
        pass

    def augment(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise self.augment(args, kwargs)


class DataAugmentation(Augmentation):
    r"""Base class for data augmentor."""
    def __init__(self, augmentors: List[Augmentation]):
        super(DataAugmentation, self).__init__()
        self.augmentors = augmentors

    def augment(self, data: Data):
        raise NotImplementedError

    def __call__(self, data: Data):
        raise self.augment(data=data)
