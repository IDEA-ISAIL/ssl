from typing import Type
from torch_geometric.typing import *
from src.augment import Augmentor, AugmentorList, AugmentorDict

AugmentType = Union[Type[Augmentor], Type[AugmentorList], Type[AugmentorDict]]
OptAugment = Optional[AugmentType]
