"""Shared training loop for the neural networks (FFNN and LSTM).

    Returns the trained network and the best epoch,
    i.e. the epoch with the lowest validation loss, or simply the max epochs when no validation set is given.
"""
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def to_tensor(array) -> torch.Tensor:
    """NumPy/pandas values -> float32 tensor (PyTorch's default precision for weights)."""
    return torch.as_tensor(np.asarray(array, dtype=np.float32))


def train_network(net: nn.Module, X_train, y_train: torch.Tensor,
                  X_val=None, y_val: torch.Tensor | None = None, *,
                  learning_rate: float, batch_size: int, epochs: int, patience: int
                  ) -> tuple[nn.Module, int]:
    """Train `net` to predict y from X with mean squared error and the Adam optimiser.

    X can be one tensor (FFNN) or a tuple of tensors (LSTM: sequence and static inputs);

    """
    multi_input = isinstance(X_train, tuple) # determines how to unpack the batches from the DataLoader based on whether the network has one (FFNN) or two (LSTM) inputs
    train_inputs = X_train if multi_input else (X_train,)

    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Training rows are shuffled every epoch (so batches differ)
    generator = torch.Generator().manual_seed(torch.initial_seed())
    loader = DataLoader(TensorDataset(*train_inputs, y_train), batch_size=batch_size,
                        shuffle=True, generator=generator)

    use_early_stopping = X_val is not None #early stopping can only happen if a validation set is provided, otherwise the model will train for the full number of epochs
    
    #initialise the best loss, best epoch, and best state to keep track of the best model during training
    best_loss, best_epoch, best_state = float("inf"), epochs, None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        # One pass over the training rows, in mini-batches
        net.train()
        for *x_parts, y_batch in loader:
            X_batch = tuple(x_parts) if multi_input else x_parts[0]   # same form as X_train
            optimiser.zero_grad()                       # clear gradients from the last batch
            loss = loss_fn(net(X_batch).squeeze(-1), y_batch)
            loss.backward()                             # compute gradients
            optimiser.step()                            # update the weights

        if not use_early_stopping:
            continue                                    # final run: train for the max number of epochs

        # Check validation loss (no weight updates here)
        net.eval()
        with torch.no_grad():
            val_loss = loss_fn(net(X_val).squeeze(-1), y_val).item()

        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(net.state_dict())   # remember the best weights so far
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break                                   # no improvement -> early stopping

    if best_state is not None:
        net.load_state_dict(best_state)                 # restore the best epoch's weights
    net.eval()
    return net, best_epoch


def count_parameters(net: nn.Module) -> int:
    """Number of trainable weights and biases, for the cost comparison."""
    return int(sum(p.numel() for p in net.parameters() if p.requires_grad))