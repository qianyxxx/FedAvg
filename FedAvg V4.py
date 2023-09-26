'''
Author: Yan Qian
Date: 2023-09-25 16:15:28
LastEditors: Yan Qian
LastEditTime: 2023-09-25 16:29:15
Description: With modulized functions
'''
import copy
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# 数据处理
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# 下载MNIST数据集
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)

# 创建数据加载器
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

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
        x = self.fc4(x)  # 不再使用softmax

        return x

def train(model, criterion, trainloader, epochs):
    losses = []  # 存储每一轮的损失值
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        running_loss /= len(trainloader)
        losses.append(running_loss)
        print(f"Training loss: {running_loss}")
    return model, losses

def test(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    
def main():
    # 初始化模型和损失函数
    model = Classifier()
    criterion = nn.CrossEntropyLoss()  

    # 训练模型，注意这里我们需要传入criterion
    model, losses = train(model, criterion, trainloader, 5)  

    # 测试模型
    test(model, testloader)
    
if __name__ == "__main__":
    main()
