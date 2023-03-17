import torch


def get_label_pairs(n_pos: int, n_neg: int):
    r"""Get the positive and negative files."""
    label_pos = torch.ones(n_pos)
    label_neg = torch.zeros(n_neg)
    labels = torch.stack((label_pos, label_neg))
    return labels
