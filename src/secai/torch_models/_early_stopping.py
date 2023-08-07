import torch
import numpy as np
from typing import Union
from pathlib import Path


class EarlyStopping:
    """
    Stop the training early if the validation loss
    doesn't improve after a given patience.

    Parameters
    ----------
    patience: int, default=7
        How long to wait after last time validation loss improved.
    verbose : bool, default=False
        If True, prints a message for each validation loss improvement.
    delta : float, default=0.
        Minimum change in the monitored quantity to qualify as an improvement.
    path : str or Path, default='checkpoint.pt'
        Path for the checkpoint to be saved to.
    """
    def __init__(self, patience: int = 7, verbose: bool = False,
                 delta: float = 0.,
                 path: Union[str, Path] = 'torch_mlp_model.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.epoch = np.Inf
        self.minimum_validation_loss = np.Inf
        self.delta = delta
        self.path = Path(path)

    def __call__(self, validation_loss: float, model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer, epoch: int):
        """
        Call the EarlyStopper.

        Parameters
        ----------
        validation_loss : float
            The current validation loss.
        model : torch.nn.Module
            The PyTorch model.
        optimizer : torch.optim.Optimizer
            The optimizer used to train the model.
        epoch : int
            The current training epoch.
        """
        score = -validation_loss
        if self.best_score is None:
            self.best_score = score
            self.epoch = epoch
            self.save_checkpoint(validation_loss, model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of '
                  f'{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.epoch = epoch
            self.save_checkpoint(validation_loss, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, validation_loss: float, model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer):
        """
        Save the model when validation loss decreases.

        Parameters
        ----------
        validation_loss : float
            The current validation loss
        model : torch.nn.Module
            The PyTorch model.
        optimizer : torch.optim.Optimizer
            The optimizer used to train the model.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.minimum_validation_loss} '
                  f'--> {validation_loss}). Saving model ...')
        torch.save({'epoch': self.epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'validation_loss': validation_loss}, self.path)
        self.minimum_validation_loss = validation_loss
