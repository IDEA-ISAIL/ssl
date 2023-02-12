import torch

from torch_geometric.typing import Tensor, OptTensor


class DiscriminatorDGI(torch.nn.Module):
    def __init__(self, dim_h: int = 512):
        super().__init__()
        self.f_k = torch.nn.Bilinear(dim_h, dim_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c: Tensor, h_pl: Tensor, h_mi: Tensor, s_bias1: OptTensor = None, s_bias2: OptTensor = None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)
        return logits

class DiscriminatorMVGRL(torch.nn.Module):
    def __init__(self, dim_h: int = 512):
        super().__init__()
        self.f_k = torch.nn.Bilinear(dim_h, dim_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c_1: Tensor, c_2: Tensor, h_1: Tensor, h_2: Tensor, h_3: Tensor, h_4: Tensor,
                s_bias1: OptTensor = None, s_bias2: OptTensor = None):
        c_x1 = torch.unsqueeze(c_1, 1)
        c_x1 = c_x1.expand_as(h_1).contiguous()

        c_x2 = torch.unsqueeze(c_2, 1)
        c_x2 = c_x2.expand_as(h_2).contiguous()

        sc_1 = torch.squeeze(self.f_k(h_2, c_x1), 2)
        sc_2 = torch.squeeze(self.f_k(h_1, c_x2), 2)

        sc_3 = torch.squeeze(self.f_k(h_4, c_x1), 2)
        sc_4 = torch.squeeze(self.f_k(h_3, c_x2), 2)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)
        return logits