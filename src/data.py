'''
Author: Yan Qian
Date: 2023-09-27 10:25:43
LastEditors: Yan Qian
LastEditTime: 2023-10-03 11:34:03
Description: Do not edit
'''

from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Subset

def load_data(dataset='MNIST'):
    """
    Function to load the required dataset.
    Can load either MNIST or CIFAR10 dataset based on the parameter.
    Returns trainset and testset.
    """
    transform_mnist = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))])  # MNIST 只有一个颜色通道

    transform_cifar = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # CIFAR10 and CIFAR100 have three color channels, so there are three groups of parameters

    if dataset == 'MNIST':
        # Load the training data
        trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform_mnist)
        # Load the testing data
        testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform_mnist)
    elif dataset == 'CIFAR10':
        # Load the training data
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
        # Load the testing data
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
    elif dataset == 'CIFAR100':   #Add this condition
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_cifar)
        testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_cifar)
    else:
        raise ValueError(f"Invalid dataset name. Expected 'MNIST', 'CIFAR10' or 'CIFAR100', but got '{dataset}'")

    return trainset, testset

def non_iid_partition(dataset, num_clients):
    # Partition dataset to simulate non-iid data among clients.
    # For simplicity, assume each client gets data of only 1 or 2 classes.
    # Get the labels
    labels = [target for _, target in dataset]
    
    # Split the indices by label
    label_indices = [[] for _ in range(10)]  # assuming 10 classes for simplicity
    for idx, label in enumerate(labels):
        label_indices[label].append(idx)
    
    # Distribute the label indices among clients
    num_labels_per_client = 10 // num_clients
    client_data = {}
    for i in range(num_clients):
        start_label = i * num_labels_per_client
        end_label = start_label + num_labels_per_client
        indices = []
        for j in range(start_label, end_label):
            indices.extend(label_indices[j])
        
        client_data[i] = indices
        
    return client_data
