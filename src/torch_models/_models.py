"""Various PyTorch models."""
from torch import nn


class LinearRegression(nn.Module):
    def __init__(self, in_features, num_classes):
        super(LinearRegression, self).__init__()
        self.linear_layer = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.linear_layer(x)
        return x


class MultilayerPerceptron(nn.Module):
    def __init__(self, in_features, num_classes):
        super(MultilayerPerceptron, self).__init__()
        self.linear_layer = nn.Linear(in_features, num_classes)
        self.non_linearity = nn.ReLU()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.linear_layer(x)
        x = self.non_linearity(x)
        return x
