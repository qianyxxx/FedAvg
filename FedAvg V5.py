# Import necessary libraries
import copy                    # For creating deep copies of data structures
import torch                   # Fundamental package for scientific computing with PyTorch
import matplotlib.pyplot as plt  # For creating visualizations
from torch import nn, optim    # Importing necessary functionalities from PyTorch
from torchvision import datasets, transforms # For loading datasets and performing transformations
import pandas as pd
from tqdm import tqdm

# Function to load data
def load_data():
    # Applying transformations to the dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # Load the MNIST training dataset
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    # Load the MNIST testing dataset
    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    return trainset, testset

# Defining the Neural Network model (Classifier)
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers in the network
        self.fc1 = nn.Linear(784, 256)   # First fully connected layer
        self.fc2 = nn.Linear(256, 128)   # Second fully connected layer
        self.fc3 = nn.Linear(128, 64)    # Third fully connected layer
        self.fc4 = nn.Linear(64, 10)     # Last fully connected layer

    # Define the forward pass
    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten the input tensor
        x = torch.relu(self.fc1(x)) # Pass through first layer and activation function
        x = torch.relu(self.fc2(x)) # Pass through second layer and activation function
        x = torch.relu(self.fc3(x)) # Pass through third layer and activation function
        x = torch.log_softmax(self.fc4(x), dim=1) # Pass through final layer and log softmax function
        return x

# Function that trains the model
def train(model, trainloader, epochs=5):
    losses = [] 
    optimizer = optim.Adam(model.parameters(), lr=0.003) 
    criterion = nn.CrossEntropyLoss() 
    for e in tqdm(range(epochs), desc = "Training"):  # 使用 tqdm 插件显示进度
        running_loss = 0
        for images, labels in trainloader:
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

# Function to train model on different clients
def client_train(model, trainset, clients=10, epochs=5):
    client_data = list(torch.utils.data.random_split(trainset, [int(trainset.data.shape[0] / clients) for _ in range(clients)]))
    client_models = []    # Stores the models of each client
    client_losses = []    # Stores the losses of each client
    # Train each client model
    for i in range(clients):
        print(f"Training on client {i+1}...")
        client_model, client_loss = train(copy.deepcopy(model), torch.utils.data.DataLoader(client_data[i]), epochs)
        client_models.append(client_model)
        client_losses.append(client_loss) 
    return client_models, client_losses

# Function to average the models from the clients
def avg_model(client_models):
    avg_state_dict = copy.deepcopy(client_models[0].state_dict())
    for key in avg_state_dict.keys():
        for i in range(1, len(client_models)):
            avg_state_dict[key] += client_models[i].state_dict()[key]
        avg_state_dict[key] = torch.div(avg_state_dict[key], len(client_models))
    model.load_state_dict(avg_state_dict)
    return model

# Function to evaluate the model
def evaluate(model, testloader):
    correct = 0
    total = 0
    # Loop through all batches in the test loader
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    return 100 * correct / total  # 返回精度值以供后续使用

# Main function
def main():
    model = Classifier()  
    trainset, testset = load_data()
    client_models, client_losses, accuracies = [], [], []  # 增加列表来存储每个客户端的精度值
    for i in range(10):  # 假设有10个客户端
        client_model, client_loss = client_train(model, trainset)
        client_models.append(client_model)
        client_losses.append(client_loss)
    model_avg = avg_model(client_models)  # 先定义和求解 model_avg
    for i in range(10):  # 假设有10个客户端
        accuracy = evaluate(model_avg, torch.utils.data.DataLoader(testset))  # 然后在此使用 model_avg
        accuracies.append(accuracy)
    
    # 绘制精度图表
    plt.figure(figsize=(10, 7))
    plt.plot(range(10), accuracies, marker='o')
    plt.title('Test Accuracy of Clients')
    plt.xlabel('Client')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.show()

    # 将数据写入Excel文件
    df = pd.DataFrame({
        'Client': range(10),
        'Test Accuracy': accuracies
    })
    df.to_excel('./results.xlsx', index=False)

if __name__ == "__main__":
    main()

