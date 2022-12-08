

class Data:
    """
    TODO: torch geomtric data structure.
    """
    def load(self, **kwargs) -> None:
        raise NotImplementedError


class HomoData(Data):
    def __init__(self):
        self.adj = None
        self.attrs = None

    def load(self, path: str) -> None:
        """
        Load the specified dataset.
        path: the folder of the dataset.
        """
        raise NotImplementedError

    @property
    def n_nodes(self):
        return len(self.attrs)
