import copy
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# 处理数据集
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# 下载MNIST数据集
dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)

# 创建数据加载器
trainloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# 创建模型
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

# 初始化模型并设置优化器和损失函数
model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# 训练函数
def train(model, trainloader, epochs):
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    return model

# 客户端数量和训练周期
clients = 10
epochs = 5

# 简单模拟数据在不同客户端上的分布
client_data = list(torch.utils.data.random_split(dataset, [int(dataset.data.shape[0] / clients) for _ in range(clients)]))

client_models = []
# 在每个客户端上训练模型
for i in range(clients):
    client_model = train(copy.deepcopy(model), torch.utils.data.DataLoader(client_data[i]), epochs)
    client_models.append(client_model)

# 聚合所有客户端的模型
avg_state_dict = copy.deepcopy(client_models[0].state_dict())
for key in avg_state_dict.keys():
    for i in range(1, len(client_models)):
        avg_state_dict[key] += client_models[i].state_dict()[key]
    avg_state_dict[key] = torch.div(avg_state_dict[key], len(client_models))

# 加载平均模型
model.load_state_dict(avg_state_dict)