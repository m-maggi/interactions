# taken from https://github.com/neeland/local-glm-net and slightly modified

import torch
from torch import nn


class LocalGLMNet(nn.Module):
    """
    A neural network model for generalized linear models with local linear
    approximation. The model consists of a series of fully connected hidden
    layers with a tanh activation function, followed by a skip connection and
    an output layer with an exponential activation function. The skip connection
    is computed as the dot product between the output of the last hidden layer
    and the input features. The output of the model is the element-wise product
    of the output of the output layer and an exposure parameter (if provided).
    """

    def __init__(self, input_size):
        super(LocalGLMNet, self).__init__()

        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, 80)])
        self.hidden_layers.extend(
            [
                nn.Linear(80, 40),
                nn.Linear(40, 30),
            ]
        )
        self.last_hidden_layer = nn.Linear(30, input_size)
        self.output_layer = nn.Linear(1, 1)
        self.activation = nn.Tanh()
        self.inverse_link = torch.exp

    def forward(self, features, exposure=None, attentions=False):
        """
        Forward pass of the model.

        Args:
            features (torch.Tensor): Input features.
            exposure (torch.Tensor, optional): Exposure variable. Defaults to None.
            attentions (bool, optional): Whether to return attention weights. Defaults to False.

        Returns:
            torch.Tensor: Model output.
        """
        x = features
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.last_hidden_layer(x)
        if attentions:
            return x
        # Dot product
        skip_connection = torch.einsum("ij,ij->i", x, features).unsqueeze(1)
        x = self.output_layer(skip_connection)
        x = self.inverse_link(x)
        if exposure is None:
            exposure = torch.ones_like(x, device=features.device)
        x = x * exposure
        return x
