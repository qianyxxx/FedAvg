import torch
from torch import nn, optim
from tqdm import tqdm
import copy
import torch.multiprocessing as mp
from model import Classifier
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

def client_train(client_id, trainset, model, epochs, lr, batch_size, queue, device='cuda'):
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

def global_train(model, trainset, testset, rounds, epochs, lr, batch_size, num_clients, device='cuda'):
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
        global_model = avg_model(client_models, device)
        # Evaluate the achieved accuracies
        accuracy = evaluate(global_model, torch.utils.data.DataLoader(testset), device)
        global_accuracies.append(accuracy)
    return global_accuracies

def avg_model(client_models,device):
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
