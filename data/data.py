from typing import Optional
from torch_geometric.typing import Adj, OptTensor


class Data:
    """
    TODO: geometric data structure
    """
    def __init__(self,
                 x: OptTensor = None,
                 adj: Optional[Adj] = None,
                 **kwargs):
        if x is not None:
            self.x = x
        if adj is not None:
            self.adj = adj

    @property
    def n_nodes(self) -> int:
        raise NotImplementedError


class HomoData(Data):
    def __init__(self, x: OptTensor = None, adj: Optional[Adj] = None):
        super().__init__(x=x, adj=adj)

    @property
    def n_nodes(self) -> int:
        return len(self.adj)
