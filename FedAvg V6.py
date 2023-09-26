# Import necessary libraries
import copy   
import torch   
import matplotlib.pyplot as plt  
from torch import nn, optim    
from torchvision import datasets, transforms 
import pandas as pd
from tqdm import tqdm

# Define the number of clients as a global variable at the top of your script
NUM_CLIENTS = 10

# 检查是否有 CUDA 设备可用，如果有，我们将使用 GPU 来加快训练速度，否则我们将使用 CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Function to load data
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
        x = torch.log_softmax(self.fc4(x), dim=1)
        return x

# Function that trains the model
def train(model, trainloader, epochs=5):
    model.to(device)  # Send model to GPU
    # # Print the device that will be used for training
    # print(f'Training on device: {device}')
    
    losses = [] 
    optimizer = optim.Adam(model.parameters(), lr=0.003) 
    criterion = nn.CrossEntropyLoss() 
    for e in tqdm(range(epochs), desc = "Training"):
        running_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)  # Send inputs and labels to GPU
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        running_loss /= len(trainloader)
        print(f"Epoch {e+1}/{epochs} - Loss: {running_loss}")
        losses.append(running_loss)
    return model, losses

def client_train(model, trainset, epochs=5):
    client_data = list(torch.utils.data.random_split(trainset, [int(trainset.data.shape[0] / NUM_CLIENTS) for _ in range(NUM_CLIENTS)]))
    client_models = []
    client_losses = []
    for i in range(NUM_CLIENTS):
        print(f"Training on client {i+1}...")
        client_model, client_loss = train(copy.deepcopy(model), torch.utils.data.DataLoader(client_data[i]), epochs)
        client_models.append(client_model)
        client_losses.append(client_loss) 
    return client_models, client_losses

def avg_model(client_models):
    model = Classifier()  # 创建一个新的 model 的实例
    model.to(device)  # 确保新模型在相同的设备（CPU或GPU）
    avg_state_dict = copy.deepcopy(client_models[0].state_dict())
    for key in avg_state_dict.keys():
        for i in range(1, len(client_models)):
            avg_state_dict[key] += client_models[i].state_dict()[key]
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

# Main function
# Main function
def main():
    # Print the device that will be used for training
    print(f'Training on device: {device}')

    trainset, testset = load_data()
    model = Classifier()
    model.to(device)  # Send model to GPU

    client_models, client_losses = client_train(model, trainset)

    accuracies = []

    for i, client_model in enumerate(client_models):
        print(f"Evaluating client {i+1} model...")
        accuracy = evaluate(client_model, torch.utils.data.DataLoader(testset))
        accuracies.append(accuracy)

    plt.figure(figsize=(10, 7))
    plt.plot(range(NUM_CLIENTS), accuracies, marker='o')
    plt.title('Test Accuracy of Clients')
    plt.xlabel('Client')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.show()

    df = pd.DataFrame({
        'Client': range(NUM_CLIENTS),  # 这里的数字10更改为了 NUM_CLIENTS
        'Test Accuracy': accuracies
    })
    df.to_excel('./results.xlsx', index=False)

if __name__ == "__main__":
    main()
