'''
Author: Yan Qian
Date: 2023-09-27 10:31:59
LastEditors: Yan Qian
LastEditTime: 2023-09-27 11:46:51
Description: Do not edit
'''
import torch

def evaluate(model, testloader, device):
    """
    Function to evaluate the model.
    :param model: Model to be evaluated
    :param testloader: Loader for the testing data
    """
    #...
    # Prepare the model for evaluation
    if model is None:
        print("Model is None.")
        return 0  # Return a default value or otherwise handle this situation
    else:
        model.eval()
        model.to(device) 
    print(f'Testing on device: {device}')

    correct = 0
    total = 0
    # No gradient computation while evaluating
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total
