'''
Author: Yan Qian
Date: 2023-09-27 10:25:43
LastEditors: Yan Qian
LastEditTime: 2023-09-27 12:07:18
Description: Do not edit
'''
from torchvision import datasets, transforms

def load_data(dataset='MNIST'):
    """
    Function to load the required dataset.
    Can load either MNIST or CIFAR10 dataset based on the parameter.
    Returns trainset and testset.
    """
    transform_mnist = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))])  # MNIST 只有一个颜色通道

    transform_cifar10 = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # CIFAR10 有三个颜色通道，所以有三组参数

    if dataset == 'MNIST':
        # Load the training data
        trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform_mnist)
        # Load the testing data
        testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform_mnist)
    elif dataset == 'CIFAR10':
        # Load the training data
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar10)
        # Load the testing data
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar10)
    else:
        raise ValueError(f"Invalid dataset name. Expected 'MNIST' or 'CIFAR10', but got '{dataset}'")

    return trainset, testset

