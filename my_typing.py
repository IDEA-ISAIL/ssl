""" Next update. """

from typing import Union, Optional
from augment import Augmentor, AugmentorList, AugmentorDict

AugmentType = Union[Augmentor, AugmentorList, AugmentorDict]
OptAugment = Optional[AugmentType]
