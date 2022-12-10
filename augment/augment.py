import logging


__all__ = [
    "Augment"
]


class Augment:
    def positive(self, *args, **kwargs):
        """
        Positive augmentation.
        """
        raise NotImplementedError

    def negative(self, *args, **kwargs):
        """
        Negative augmentation.
        """
        raise NotImplementedError
