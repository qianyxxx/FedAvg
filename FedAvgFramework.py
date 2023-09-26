# Import the libraries
import argparse
import copy
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp

# Set the start method for multiple processes
mp.set_start_method('spawn', force=True)

# Check if CUDA is available, else use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


def train(model, trainloader, epochs, lr, pbar=None):
    """
    Function to train the model.
    :param model: neural network model to be trained
    :param trainloader: the loader for the training data
    :param epochs: the number of training epochs
    :param lr: learning rate
    :param batch_size: size of batches for training
    :param pbar: progress bar
    """
    # Prepare the model for training
    model.train()
    # Move the model to GPU
    model.to(device)  
    losses = []
    # Define the optimizer with specified learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Define the loss criterion
    criterion = nn.CrossEntropyLoss()

    # Initialize the progress bar
    pbar = tqdm(total=epochs, desc="Training")
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Move the input and label to GPU
            images, labels = images.to(device), labels.to(device)  
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            # Clip the gradient to prevent exploding gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
            optimizer.step()
            running_loss += loss.item()

        running_loss /= len(trainloader)

        # Update the progress bar
        pbar.set_postfix({'Loss': running_loss})
        pbar.update()
        losses.append(running_loss)

    # Close the progress bar
    pbar.close()
    return model, losses


def client_train(client_id, trainset, model, epochs, lr, batch_size, queue):
    """
    Function to train clients.
    :param client_id: ID of the client
    :param trainset: Training data
    :param model: The model to train
    :param epochs: Number of epochs for training
    :param lr: Learning rate
    :param batch_size: Size of the training batch
    :param queue: Queue used for interprocess communication
    """
    with tqdm(total=100, desc=f"Client {client_id+1}", unit='batch') as pbar:
        try:
            client_model, client_loss = train(copy.deepcopy(model).cpu(), torch.utils.data.DataLoader(trainset, batch_size=batch_size), epochs=epochs, lr=lr)
            client_model = client_model.to(device)
            if client_model is not None:
                queue.put((client_id, client_model.cpu().state_dict(), client_loss))
            else:
                queue.put((client_id, None, None))
        except Exception as e:
            queue.put((client_id, None, None))
        pbar.update()


def parallel_clients_train(model, trainset, epochs, lr, batch_size, num_clients):
    """
    Function to train clients in parallel.
    :param model: The original neural network model
    :param trainset: Training dataset
    :param epochs: Number of epochs for training
    :param lr: Learning rate
    :param batch_size: Size of the training batch
    :param num_clients: Number of clients
    """
    # Split the dataset among clients
    client_data = list(torch.utils.data.random_split(trainset, [int(trainset.data.shape[0] / num_clients) for _ in range(num_clients)]))
    client_models = [None]*num_clients
    client_losses = [None]*num_clients

    processes = []
    queue = mp.Manager().Queue()
    with tqdm(total=num_clients, desc="Clients", unit='client'):
        for i in range(num_clients):
            p = mp.Process(target=client_train, args=(i, client_data[i], model, epochs, lr, batch_size, queue)) 
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        while not queue.empty():
            client_id, client_model_state, client_loss = queue.get()
            if client_model_state is not None:
                client_model = Classifier()
                client_model.load_state_dict(client_model_state)
                client_models[client_id] = client_model
                client_losses[client_id] = client_loss
            else:
                print(f"Client {client_id} did not return a valid model.")

    return client_models, client_losses

def global_train(model, trainset, testset, rounds, epochs, lr, batch_size, num_clients):
    """
    Function to perform global training.
    :param model: The original neural network model
    :param trainset: Training dataset
    :param testset: Testing dataset
    :param rounds: Number of communication rounds
    :param epochs: Number of epochs for training
    :param lr: Learning rate
    :param batch_size: Size of the training batch
    :param num_clients: Number of clients
    """
    global_model = model
    global_accuracies = []
    for round in range(rounds):
        print(f"Communication round {round+1}/{rounds}...")
        # Perform client training in parallel
        client_models, _ = parallel_clients_train(global_model, trainset, epochs, lr, batch_size, num_clients)
        # Average the client models
        global_model = avg_model(client_models)
        # Evaluate the achieved accuracies
        accuracy = evaluate(global_model, torch.utils.data.DataLoader(testset))
        global_accuracies.append(accuracy)
    return global_accuracies


def avg_model(client_models):
    """
    Function to average the model parameters.
    :param client_models: List of client models
    """
    model = Classifier()
    model.to(device)
    avg_state_dict = {}
    # Average the model parameters
    for key in client_models[0].state_dict().keys():
        avg_state_dict[key] = sum([client_model.state_dict()[key] for client_model in client_models])
        avg_state_dict[key] = torch.div(avg_state_dict[key], len(client_models))
    model.load_state_dict(avg_state_dict)
    return model


def evaluate(model, testloader):
    """
    Function to evaluate the model.
    :param model: Model to be evaluated
    :param testloader: Loader for the testing data
    """
    # Prepare the model for evaluation
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


def main(args):
    """
    Main function for executing the program.
    """
    # Get command line arguments
    lr = args.lr if args.lr else 0.01
    num_clients = args.num_clients if args.num_clients else 1
    epochs = args.epochs if args.epochs else 5
    batch_size = args.batch_size if args.batch_size else 64
    rounds = args.rounds if args.rounds else 10

    if not lr or not num_clients or not epochs or not batch_size or not rounds:
        print("All arguments must be specified. Add '-h' for help.")
        return

    print(f'Training on device: {device}')

    trainset, testset = load_data()
    model = Classifier()
    model.to(device)

    # Perform global training
    global_accuracies = global_train(model, trainset, testset, rounds=rounds, epochs=epochs, lr=lr, batch_size=batch_size, num_clients=num_clients)
    # Plot the resulting accuracies over the communication rounds
    fig, ax = plt.subplots(figsize=(12, 8))

    # 制作一些样本数据
    rounds_for_plot = np.array(range(1, rounds+1))
    accuracy = np.array(global_accuracies)

    # 绘制精度随通信轮次变化的曲线
    ax.plot(rounds_for_plot, accuracy, label='Test Accuracy', marker='o')

    # 设置标题和坐标轴标签
    ax.set_title('Test Accuracy over Communication Rounds', fontsize=14)
    ax.set_xlabel('Communication Rounds', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)

    # 添加网格
    ax.grid(True)

    # 添加图例
    ax.legend(fontsize=12)

    # 自动调整子图参数，这使得子图适应figure区域
    plt.tight_layout()

    # 保存图片
    plt.savefig('global_results.png')

    # Save the results in a dataframe
    df = pd.DataFrame({
        'Communication Round': np.arange(1, rounds+1).tolist(),
        'Test Accuracy': global_accuracies
    })
    df.to_excel('./global_results.xlsx', index=False)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--num_clients', type=int, help='Number of clients')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--rounds', type=int, help='Number of communication rounds')
    args = parser.parse_args()

    # Call the main function
    main(args)
