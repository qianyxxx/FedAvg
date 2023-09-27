'''
Author: Yan Qian
Date: 2023-09-27 10:27:59
LastEditors: Yan Qian
LastEditTime: 2023-09-27 10:28:04
Description: Do not edit
'''
import torch
from torch import nn

class Classifier(nn.Module):
    """
    Define a neural network model for the classification of the MNIST dataset.
    """
    def __init__(self):
        super().__init__()
        # Define the layers in the model
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        """
        Method to define the forward pass.
        :param x: input tensor
        :return: output tensor
        """
        # Flatten the input tensor
        x = x.view(x.shape[0], -1)
        # Define the activation functions for the layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
