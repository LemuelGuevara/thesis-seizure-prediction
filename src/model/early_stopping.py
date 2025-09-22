import os

import torch


class EarlyStopping:
    def __init__(self, path, patience=7, min_delta=0, verbose=False):
        """
        Args:
            path (str): Full path (including filename) to save the checkpoint.
            patience (int): Early stopping patience.
            min_delta (float): Minimum improvement to reset patience.
            verbose (bool): Verbosity.
        """
        # ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.path = path
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
