import torch

from typing import Optional
from src.typing import OptAugment


ENCODER_NAME = "encoder.ckpt"
MODEL_NAME = "model.ckpt"


class Method(torch.nn.Module):
    r"""Base class for self-supervised learning methods.

    Args:
        encoder (torch.nn.Module): the encoder to be trained.
        save_root (str): the root to save the model/encoder.
        use_cuda (bool): whether to use cuda or not.
        *args:
        **kwargs:
    """
    def __init__(
            self,
            encoder: torch.nn.Module,
            data_augment: OptAugment = None,
            emb_augment: OptAugment = None,
            use_cuda: bool = True,   # TODO: check if this arg is useful?
            *args,
            **kwargs):
        super().__init__()

        self.encoder = encoder  # entire model to train, including encoders and other necessary modules
        self.data_augment = data_augment
        self.emb_augment = emb_augment
        self.use_cuda = use_cuda

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def forward(self, data):
        r"""Perform the forward pass."""
        raise NotImplementedError

    def train_iter(self, *args, **kwargs):
        r"""Each training iteration."""
        raise NotImplementedError

    def get_embs(self, data, is_numpy: bool = False, *args, **kwargs):
        embs = self.encoder(data, *args, **kwargs).detach()
        if is_numpy:
            return embs.cpu().numpy()
        return embs

    def save_encoder(self, path: Optional[str] = None) -> None:
        r"""Only save the parameters of the encoder.

        Args:
            path (Optional[str]): path to save the parameters.
        """
        if path is None:
            path = ENCODER_NAME
        torch.save(self.encoder.state_dict(), path)

    def save(self, path: Optional[str] = None) -> None:
        r"""Save the parameters of the entire model.

        Args:
            path (Optional[str]): path to save the parameters.
        """
        if path is None:
            path = MODEL_NAME
        torch.save(self.state_dict(), path)

    def load_encoder(self, path: str) -> None:
        r"""Only load the parameters of the encoder."""
        state_dict = torch.load(path)
        self.encoder.load_state_dict(state_dict)

    def load(self, path: str) -> None:
        r"""Load the parameters of the entire model."""
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
