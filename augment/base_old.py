import copy
from data import Data

from typing import List, Union, Dict


__all__ = [
    "Augmentor",
    "Augmentors",
    "Augmentation",
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


class Augmentors:
    r"""A wrapper for a list of augmentors."""
    def __init__(self, augmentors: Union[Augmentor, List[Augmentor]]):
        if type(augmentors) == List:
            self.augmentors = augmentors
        else:
            self.augmentors = [augmentors]

    def apply(self, inputs):
        r"""Apply the augmentors to inputs."""
        for augmentor in self.augmentors:
            inputs = augmentor(inputs)
        return inputs

    def __call__(self, inputs):
        self.apply(inputs)


class Augmentation:
    r"""Base class for augmentation. A dictionary of augmentors."""
    def __init__(self, augment_dict: Dict[str, Augmentors]):
        self.augment_dict = augment_dict

    def apply(self, data: Data):
        data_tmp = copy.deepcopy(data)
        for augmentor in self.augmentors:
            data_tmp = augmentor(data_tmp)
        return data_tmp

    def __call__(self, data: Data):
        return self.apply(data)
