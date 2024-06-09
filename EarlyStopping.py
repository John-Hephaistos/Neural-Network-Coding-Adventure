import copy


class EarlyStopping:
    def __init__(self, patience = 5, delta = 0, restore_best_weights = True):
        self._patience = patience
        self._delta = delta
        self._restore_best_weights = restore_best_weights
        self.best_loss = None
        self.best_model = None
        self.counter = 0
        self.status = ""


    def check(self, model, val_loss):

        # Initialize
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
        elif val_loss - self.best_loss >= self._delta:
            self.best_loss = val_loss
            self.counter = 0
        elif val_loss - self.best_loss < self._delta:
            self.counter += 1
            if self.counter >= self._patience:
                self.status = f"STOPPED on {self.counter}, current best loss: {self.best_loss}"
                if self._restore_best_weights:
                    model.load_state_dict(self.best_model.state_dict())
                return True
        self.status = f"{self.counter} // {self._patience}"
        print(self.status)
        return False
