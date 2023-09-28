import torch
from torch import nn, optim
from tqdm import tqdm
import copy
import torch.multiprocessing as mp
from model import MNIST_Classifier, CIFAR10_Classifier, CIFAR100_Classifier
from evaluate import evaluate

def train(model, trainloader, epochs, lr, pbar=None, device='cuda'):
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
    if pbar is None:
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

def client_train(client_id, trainset, global_model_dict, epochs, lr, batch_size, queue, dataset, device='cuda'):
    """
    Function to train clients.
    :param client_id: ID of the client
    :param trainset: Training data
    :param global_model_dict: The state dict of the global model
    :param epochs: Number of epochs for training
    :param lr: Learning rate
    :param batch_size: Size of the training batch
    :param queue: Queue used for interprocess communication
    """
    with torch.cuda.device(device):  # 增加上下文管理器来清理GPU内存
        with tqdm(total=100, desc=f"Client {client_id+1}", unit='batch') as pbar:
            try:
                if dataset == 'MNIST':
                    client_model = MNIST_Classifier()
                elif dataset == 'CIFAR10':
                    client_model = CIFAR10_Classifier()
                elif dataset == 'CIFAR100':  # 为CIFAR100数据集指定模型
                    client_model = CIFAR100_Classifier() 

                # Load the state dict of the global model
                client_model.load_state_dict(global_model_dict)

                client_model, client_loss = train(client_model, torch.utils.data.DataLoader(trainset, batch_size=batch_size), epochs=epochs, lr=lr)
                if client_model is not None:
                    queue.put((client_id, client_model.cpu().state_dict(), client_loss)) # 将模型移动到CPU，释放GPU内存
                else:
                    queue.put((client_id, None, None))
            except Exception as e:
                print(f"Error training client {client_id}: {e}")
                queue.put((client_id, None, None))

            pbar.update()
        # 清空所有CUDA缓存
        torch.cuda.empty_cache()
        

def parallel_clients_train(model, trainset, epochs, lr, batch_size, num_clients, dataset):
    """
    Function to train clients in parallel.
    :param model: The original neural network model
    :param trainset: Training dataset
    :param epochs: Number of epochs for training
    :param lr: Learning rate
    :param batch_size: Size of the training batch
    :param num_clients: Number of clients
    """
    # 就像在client_train中做的那样，我们需要为新数据集在这里添加一个判断条件，并据此加载适当的模型
    if model is None:
        if dataset == 'MNIST':
            model = MNIST_Classifier()
        elif dataset == 'CIFAR10':
            model = CIFAR10_Classifier()
        elif dataset == 'CIFAR100':  # 为CIFAR100数据集指定模型
            model = CIFAR100_Classifier()
        else:
            print("Invalid dataset!")
            return None

    client_data = list(torch.utils.data.random_split(trainset, [int(trainset.data.shape[0] / num_clients) for _ in range(num_clients)]))
    client_models = [None]*num_clients
    client_losses = [None]*num_clients

    processes = []
    queue = mp.Manager().Queue()
    with tqdm(total=num_clients, desc="Clients", unit='client') as pbar:
        for i in range(num_clients):
            p = mp.Process(target=client_train, args=(i, client_data[i], model.state_dict(), epochs, lr, batch_size, queue, dataset)) 
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        while not queue.empty():
            client_id, client_model_state, client_loss = queue.get()
            if client_model_state is not None:
                if dataset == 'CIFAR10':
                    client_model = CIFAR10_Classifier()
                elif dataset == 'CIFAR100':
                    client_model = CIFAR100_Classifier()
                elif dataset == 'MNIST': # 'MNIST'
                    client_model = MNIST_Classifier()
                client_model.load_state_dict(client_model_state)
                client_models[client_id] = client_model
                client_losses[client_id] = client_loss
            else:
                print(f"Client {client_id} did not return a valid model.")

    return client_models, client_losses


def global_train(model, trainset, testset, rounds, epochs, lr, batch_size, num_clients, dataset, device='cuda'):
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
    global_model = copy.deepcopy(model)
    global_accuracies = []
    for round in range(rounds):
        print(f"Communication round {round+1}/{rounds}...")
        # Perform client training in parallel
        client_models, _ = parallel_clients_train(global_model, trainset, epochs, lr, batch_size, num_clients, dataset)
        # Average the client models
        global_model = avg_model(client_models, dataset, device)
        global_model = global_model.cpu()  # Add this line
        # Evaluate the achieved accuracies
        accuracy = evaluate(global_model, torch.utils.data.DataLoader(testset), device)
        global_accuracies.append(accuracy)
    return global_accuracies

def avg_model(client_models, dataset, device='cuda'):
    """
    Function to average the model parameters.
    :param client_models: List of client models
    """
    # 这里需要为新数据集添加一个判断条件，并据此加载适当的模型
    if dataset == 'MNIST':
        model = MNIST_Classifier()
    elif dataset == 'CIFAR10':
        model = CIFAR10_Classifier()
    elif dataset == 'CIFAR100':  # 为CIFAR100数据集指定模型
        model = CIFAR100_Classifier()

    model.to(device)
    # Filter out None values from client_models
    client_models = [model for model in client_models if model is not None]
    
    if len(client_models) > 0:
        if dataset == 'MNIST':
            model = MNIST_Classifier()
        elif dataset == 'CIFAR10':
            model = CIFAR10_Classifier()
        elif dataset == 'CIFAR100':  # 为CIFAR100数据集指定模型
            model = CIFAR100_Classifier()
            
        model.to(device)
        avg_state_dict = {}
        # Average the model parameters
        for key in client_models[0].state_dict().keys():
            avg_state_dict[key] = sum([client_model.state_dict()[key] for client_model in client_models])
            avg_state_dict[key] = torch.div(avg_state_dict[key], len(client_models))
        model.load_state_dict(avg_state_dict)
        model = model.cpu()  # 将模型移动到CPU
        return model
    else:
        print("No valid client models to average.")
        return None


