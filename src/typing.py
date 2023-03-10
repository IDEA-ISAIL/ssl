from typing import Union, Optional
from src.augment import Augmentor, AugmentorList, AugmentorDict

AugmentType = Union[Augmentor, AugmentorList, AugmentorDict]
OptAugment = Optional[AugmentType]
