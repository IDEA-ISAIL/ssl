class EarlyStopper:
    def __init__(self, patience):
        self.patience = patience

        self._best_value = None
        self._n_wait = 0

    def is_stop(self, current_value):
        if self._best_value is None:
            self._best_value = current_value
            self._n_wait = 0
        else:
            if current_value < self._best_value:
                self._n_wait = 0
                return False
            else:
                self._n_wait += 1

        if self._n_wait == self.patience:
            print('Early stopped!')
            return True
