import logging


__all__ = [
    "Augment"
]


class Augment:
    def positive(self, **kwargs):
        """
        Positive augmentation.
        """
        raise NotImplementedError

    def negative(self, **kwargs):
        """
        Negative augmentation.
        """
        raise NotImplementedError
