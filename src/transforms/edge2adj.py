import torch

from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

from typing import Optional, Callable


@functional_transform('edge_to_adj_ssl')
class Edge2Adj(BaseTransform):
    r"""Convert edge to adjacency matrix.

    Args:
        norm (Optional[Callable]): Normalization of the adjacency matrix.
            (default: None)
            None: Generate 0/1 adjacency matrix from edge_index & edge_attr/edge_weight.
    """
    def __init__(self, norm: Optional[Callable] = None):
        self.norm = norm

    def __call__(self, data):
        assert 'edge_index' in data

        edge_weight = data.edge_attr
        if 'edge_weight' in data:
            edge_weight = data.edge_weight

        if self.norm is None:
            if edge_weight is None:
                data.adj = torch.sparse_coo_tensor(data.edge_index, torch.ones_like(data.edge_indx[0]),
                                                   [data.num_nodes, data.num_nodes])
            else:
                data.adj = torch.sparse_coo_tensor(data.edge_index, data.edge_weight,
                                                   [data.num_nodes, data.num_nodes])
        else:
            data = self.norm(data)
        return data
