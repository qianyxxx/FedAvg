# Import necessary libraries
import copy   
import torch   
import matplotlib.pyplot as plt  
from torch import nn, optim    
from torchvision import datasets, transforms 
import pandas as pd
from tqdm import tqdm
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)

# Define the number of clients as a global variable at the top of your script
NUM_CLIENTS = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    return trainset, testset

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256) 
        self.fc2 = nn.Linear(256, 128)  
        self.fc3 = nn.Linear(128, 64) 
        self.fc4 = nn.Linear(64, 10)  

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x)) 
        x = torch.relu(self.fc2(x)) 
        x = torch.relu(self.fc3(x)) 
        x = self.fc4(x)
        return x

# Function that trains the model
def train(model, trainloader, epochs=5, pbar=None):
    model.to(device)  # Send model to GPU
    
    losses = [] 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss() 
    
    # Initialize the progress bar
    pbar = tqdm(total=epochs, desc="Training")
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)  # Send inputs and labels to GPU
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
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


def client_train(client_id, trainset, model, queue):
    with tqdm(total=100, desc=f"Client {client_id+1}", unit='batch') as pbar:
        try:
            print(f"Training on client {client_id+1}...")
            client_model, client_loss = train(copy.deepcopy(model).cpu(), torch.utils.data.DataLoader(trainset, batch_size=32), pbar=pbar) # add batch_size parameter here
            client_model = client_model.to(device)  # change here
            if client_model is not None:
                queue.put((client_id, client_model.cpu().state_dict(), client_loss))  # change here
            else:
                print(f"Client {client_id} model is None.")
                queue.put((client_id, None, None))
        except Exception as e:
            print(f"Exception occurred while training client {client_id}: {e}")
            queue.put((client_id, None, None))
        pbar.update()



def parallel_clients_train(model, trainset, epochs=5):
    client_data = list(torch.utils.data.random_split(trainset, [int(trainset.data.shape[0] / NUM_CLIENTS) for _ in range(NUM_CLIENTS)]))
    client_models = [None]*NUM_CLIENTS
    client_losses = [None]*NUM_CLIENTS

    processes = []
    queue = mp.Manager().Queue()

    for i in range(NUM_CLIENTS):
        p = mp.Process(target=client_train, args=(i, client_data[i], model, queue))
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


    processes = []
    queue = mp.Manager().Queue()

    for i in range(NUM_CLIENTS):
        p = mp.Process(target=client_train, args=(i, client_data[i], model, queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    while not queue.empty():
        client_id, client_model_state, client_loss = queue.get()
        client_model = Classifier()
        client_model.load_state_dict(client_model_state)
        client_models[client_id] = client_model
        client_losses[client_id] = client_loss

    return client_models, client_losses


# Function for global training
def global_train(model, trainset, testset, rounds=10):
    global_model = model
    global_accuracies = []
    for round in range(rounds):
        print(f"Communication round {round+1}/{rounds}...")
        client_models, _ = parallel_clients_train(global_model, trainset)  # change function here
        global_model = avg_model(client_models)
        accuracy = evaluate(global_model, torch.utils.data.DataLoader(testset))
        global_accuracies.append(accuracy)
    return global_accuracies


def avg_model(client_models):
    model = Classifier()  # 创建一个新的 model 的实例
    model.to(device)  # 确保新模型在相同的设备（CPU或GPU）
    avg_state_dict = {}
    for key in client_models[0].state_dict().keys():
        avg_state_dict[key] = sum([client_model.state_dict()[key] for client_model in client_models])
        avg_state_dict[key] = torch.div(avg_state_dict[key], len(client_models))
    model.load_state_dict(avg_state_dict)
    return model  # 返回新的 model，而不是旧的 model



def evaluate(model, testloader):
    model.to(device)  # Send model to GPU
    # Print the device that will be used for testing
    print(f'Testing on device: {device}')
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)  # Send inputs and labels to GPU
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    return 100 * correct / total  

# 修改主函数
def main():
    # Print the device that will be used for training
    print(f'Training on device: {device}')

    trainset, testset = load_data()
    model = Classifier()
    model.to(device)  # Send model to GPU

    # Perform global training
    rounds = 10  # 设定通信轮次数，你可以根据需要更改
    global_accuracies = global_train(model, trainset, testset, rounds=rounds)

    # 绘制通信轮次与测试精度的关系图
    plt.figure(figsize=(10, 7))
    plt.plot(range(1, rounds+1), global_accuracies, marker='o')
    plt.title('Test Accuracy over Communication Rounds')
    plt.xlabel('Communication Round')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.savefig('global_results.png')  # Move this line up

    df = pd.DataFrame({
        'Communication Round': range(1, rounds+1), 
        'Test Accuracy': global_accuracies
    })
    df.to_excel('./global_results.xlsx', index=False)

if __name__ == "__main__":
    main()