from .base import Augmentor
from ssl.data import Data


class Echo(Augmentor):
    """ TODO: maybe this class is unnecessary. """
    def __init__(self):
        super().__init__()

    def apply(self, data: Data):
        return data
