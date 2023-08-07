"""Various PyTorch models."""
from torch import nn
import torch
from itertools import pairwise


class LinearRegression(nn.Module):
    """
    Linear regression model using PyTorch.

    This class contains a single linear layer to regress from the input to
    the output.

    Parameters
    ----------
    in_features : int
        Number of input features
    num_classes : int
        Number of classes, i.e., number of outputs.
    """
    def __init__(self, in_features: int, num_classes: int):
        super(LinearRegression, self).__init__()
        self.linear_layer = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor):
        """
        Forward function of the model.

        Parameters
        ----------
        x : torch.Tensor, shape=(batch_size, in_features)
            The input data.

        Returns
        -------
        x : torch.Tensor, shape=(batch_size, n_classes)
            The output data (linearly transformed input data) that approximates
            the target output.
        """
        x = x.reshape(x.shape[0], -1)
        x = self.linear_layer(x)
        return x


class MultiLayerPerceptron(nn.Module):
    """
    Train a multilayer perceptron model using PyTorch.

    This class passes the input data through an arbitrary number of hidden
    layers and always puts an activation function in between.

    The first hidden layer size needs to be the number of inputs.

    Parameters
    ----------
    hidden_layer_sizes : tuple[int, ...]
        Sizes of the different hidden layers, starting from the input size.
    num_classes : int
        Number of classes, i.e., number of outputs.
    """
    def __init__(self, hidden_layer_sizes: tuple[int, ...], num_classes: int,
                 non_linearity: nn.Module = nn.ReLU()):
        super(MultiLayerPerceptron, self).__init__()

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for in_features, out_features in pairwise(hidden_layer_sizes):
            self.hidden_layers.append(nn.Linear(in_features, out_features))

        # Output layer
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], num_classes)
        self.non_linearity = non_linearity

    def forward(self, x):
        """
        Forward function of the model.

        Parameters
        ----------
        x : torch.Tensor, shape=(batch_size, in_features)
            The input data.

        Returns
        -------
        x : torch.Tensor, shape=(batch_size, n_classes)
            The output data (non-linearly transformed input data) that
            approximates the target output.
        """
        x = x.reshape(x.shape[0], -1)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.non_linearity(x)
        x = self.non_linearity(self.output_layer(x))
        return x
