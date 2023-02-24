import copy
from data import Data

from typing import List, Union


__all__ = [
    "Augmentor",
    "Augmentation",
    "DataAugmentation"
]


class Augmentor:
    r"""Base class for augmentor."""
    def __init__(self):
        pass

    def apply(self, *args, **kwargs):
        r"""Apply the augmentor to inputs."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)


class Augmentation:
    r"""Base class for augmentation. A wrapper of augmentors."""
    def __init__(self, augmentors: Union[Augmentor, List[Augmentor]]):
        if type(augmentors) == List:
            self.augmentors = augmentors
        else:
            self.augmentors = [augmentors]

    def apply(self, *args, **kwargs):
        r"""Apply the augmentors to inputs."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class DataAugmentation(Augmentation):
    r"""Base class for data augmentation. A wrapper of augmentors."""
    def __init__(self, augmentors: Union[Augmentor, List[Augmentor]]):
        super().__init__(augmentors=augmentors)

    def apply(self, data: Data):
        data_tmp = copy.deepcopy(data)
        for augmentor in self.augmentors:
            data_tmp = augmentor(data_tmp)
        return data_tmp

    def __call__(self, data: Data):
        return self.apply(data)
