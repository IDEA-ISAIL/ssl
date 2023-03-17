class EarlyStopper:
    def __init__(self, patience, best_value):
        self.patience = patience
        self.best_value = best_value

        self.n_wait = 0

    def is_stop(self, current_value):
        if current_value < self.best_value:
            self.best_value = current_value
            self.n_wait = 0
            return False
        else:
            self.n_wait += 1

        if self.n_wait == self.patience:
            print('Early stopped!')
            return True
