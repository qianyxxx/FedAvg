'''
Author: Yan Qian
Date: 2023-09-27 10:25:43
LastEditors: Yan Qian
LastEditTime: 2023-09-27 10:27:48
Description: Do not edit
'''
import torch
from torchvision import datasets, transforms

def load_data():
    """
    Function to load the MNIST dataset.
    Returns trainset and testset.
    """
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # Load the training data
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    # Load the testing data
    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    return trainset, testset
