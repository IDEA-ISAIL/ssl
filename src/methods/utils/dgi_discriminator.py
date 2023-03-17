import torch
from src.typing import Tensor


class DGIDiscriminator(torch.nn.Module):
    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.f_k = torch.nn.Bilinear(in_channels, in_channels, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c: Tensor, h_pl: Tensor, h_mi: Tensor):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x))
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x))

        logits = torch.stack((sc_1, sc_2))
        return logits
