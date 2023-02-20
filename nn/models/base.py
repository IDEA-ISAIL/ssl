import torch


class Model(torch.nn.Module):
    r"""The full model to train the encoder.

    Args:
        encoder (torch.nn.Module): the encoder to be trained.
    """
    def __init__(self, encoder: torch.nn.Module, *args, **kwargs):
        super().__init__()
        self.encoder = encoder

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_embs(self, *args, **kwargs):
        raise NotImplementedError
