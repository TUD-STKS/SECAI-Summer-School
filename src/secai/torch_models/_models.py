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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    Multilayer perceptron model using PyTorch.

    This class passes the input data through an arbitrary number of hidden
    layers and always puts an activation function in between.

    The first hidden layer size needs to be the number of inputs.

    Parameters
    ----------
    hidden_layer_sizes : tuple[int, ...]
        Sizes of the different hidden layers, starting from the input size.
    num_classes : int
        Number of classes, i.e., number of outputs.
    non_linearity : nn.Module, default=nn.Relu()
        The activation function to be used.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class ConvolutionalNeuralNetwork(nn.Module):
    """
    Convolutional Neural Network Model using PyTorch.

    Convolve the input data with subsequent 2D filters in the following
    structure:

    The fully connected layer is required to convert the transformed image
    in a binary representation of the likelihood that it belongs to a specific
    class.

    This is hard-coded and simply serves as the starting point for own
    experiments.

    Parameters
    ----------
    in_channels : int
        Number of input channels, i.e., 1 for grayscale or 3 for RGB images.
    num_classes : int
        Number of classes, i.e., number of outputs.
    non_linearity : nn.Module, default=nn.Relu()
        The activation function to be used.
    valid : bool, default = True
        Use valid convolution or pad to equal sizes.
    """
    def __init__(self, in_channels: int, num_classes: int,
                 non_linearity: nn.Module = nn.ReLU(), valid: bool = True):
        super(ConvolutionalNeuralNetwork, self).__init__()

        # Convolution 1
        self.convolutional_layers = nn.ModuleList()

        self.convolutional_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=5,
                      stride=1, padding=0 if valid else 2),
            non_linearity,
            nn.MaxPool2d(kernel_size=2))

        # Convolution 2
        self.convolutional_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1,
                      padding=0 if valid else 2),
            non_linearity,
            nn.MaxPool2d(kernel_size=2))

        # Fully connected 1 (readout)
        self.fully_connected = nn.Linear(32 * 4 * 4 if valid else 32 * 7 * 7,
                                         num_classes)
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
        # Convolution 1
        x = self.convolutional_layer_1(x)
        # Convolution 2
        x = self.convolutional_layer_2(x)
        x = x.reshape(x.shape[0], -1)
        # Linear function (readout)
        x = self.fully_connected(x)
        x = self.non_linearity(x)
        return x


class LSTMModel(nn.Module):
    """
    Recurrent Neural Network Model with Long-Short-Term Memory cells using
    PyTorch.

    Parameters
    ----------
    input_size : int
        Number of input channels, i.e., 1 for grayscale or 3 for RGB images.
    hidden_size : int
        Number of LSTM cells in the hidden layer.
    num_layers : int
        Number of subsequent LSTM layers.
    bidirectional : bool
        Use a bidirectional LSTM model, i.e., pass the sequence forwards and
        backwards through the network.
    dropout : float
        Dropout probability for the LSTM network, i.e., the percentage of
        connections that is randomly set to zero during the training to
        prevent overfitting.
    num_classes : int
        Number of classes, i.e., number of outputs.
    non_linearity : nn.Module, default=nn.Relu()
        The activation function to be used.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 bidirectional: bool, dropout: float, num_classes: int,
                 non_linearity: nn.Module = nn.ReLU()):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)
        # Readout layer
        self.fully_connected = nn.Linear(
            2 * hidden_size if bidirectional else hidden_size, num_classes)
        self.non_linearity = non_linearity

    def forward(self, x):
        """
        Forward function of the model.

        Parameters
        ----------
        x : torch.Tensor, shape=(batch_size, n_channels, sequence_length,
        input_size)
            The input data. It is expected to be four-dimensional,
            since this model works with image data.

        Returns
        -------
        x : torch.Tensor, shape=(batch_size, n_classes)
            The output data (non-linearly transformed input data) that
            approximates the target output.
        """
        x = x.squeeze(1)
        # Initialize hidden state with zeros
        h0 = torch.zeros(
            self.num_layers * 2 if self.bidirectional else self.num_layers,
            x.shape[0], self.hidden_size).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(
            self.num_layers * 2 if self.bidirectional else self.num_layers,
            x.shape[0], self.hidden_size).requires_grad_()
        # 28 time steps
        # We need to detach as we are doing truncated
        # backpropagation through time. If we don't, we'll backprop all the way
        # to the start even after going through another batch
        x, (_, _) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # x.shape --> 100, 28, 100
        # x[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        x = self.fully_connected(x[:, -1, :])
        # x.shape --> 100, 9
        x = self.non_linearity(x)
        return x
