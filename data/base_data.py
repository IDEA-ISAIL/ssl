

class BaseData:
    """
    TODO: torch geomtric data structure.
    """
    def load(self, **kwargs):
        raise NotImplementedError


class HomoData(BaseData):
    def __init__(self):
        self.adj = None
        self.x = None       # node idx
        self.attrs = None   # node attributes

    def load(self, path: str):
        """
        Load the specified dataset.
        path: the folder of the dataset.
        """
        raise NotImplementedError
