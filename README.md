<!--
 * @Author: Yan Qian
 * @Date: 2023-09-25 17:45:49
 * @LastEditors: Yan Qian
 * @LastEditTime: 2023-09-27 12:39:35
 * @Description: Do not edit
-->
# FedAvg From Scratch

## English Version

### Federated Learning MNIST Classifier
This is a Python program using pyTorch to implement a Federated Learning system that trains an MNIST digit classifier. Clients train in parallel on their local data and the server averages the client models in each communication round to create a global model.

### Requirements

This program requires the following Python modules: argparse, copy, torch, torchvision, pandas, numpy, matplotlib, tqdm and torch.multiprocessing. Make sure that you have installed these modules on your Python environment before running the program.

### Program Description

The program begins by setting up the environment and defining the 'device' variable which will determine whether the computations will be done on the CPU or on available CUDA (GPU).
The 'load_data()' function is then defined which will load the MNIST training and testing datasets.
The 'Classifier()' class is then created which defines the structure of the neural network model used in the learning process.
Following that are functions for training individual clients ('client_train()'), training all clients in parallel ('parallel_clients_train()'), averaging client models ('avg_model()'), evaluating the model ('evaluate()'), and training the global model ('global_train()').
In the 'main()' function, the entire learning process is orchestrated.
Finally, command line arguments are parsed to adjust the learning rate, number of clients, number of epochs, batch size, and number of communication rounds.

### How to Run

To run the program, use the following command:

```
python main.py --lr 0.001 --num_clients 10 --epochs 5 --batch_size 32 --rounds 10 --dataset MNIST
```

You can adjust the learning rate, number of clients, number of epochs, batch size, number of communication rounds, and choosed dataset.

### Output

The program will output a plot visualizing the accuracy of the global model after each communication round. It also saves this data in the form of a dataframe in an Excel file named 'global_results.xlsx' and a png file named 'global_results.png'.

## Chinese Version

### 分布式学习MNIST分类器

这是使用pyTorch实现的一个Python程序，它实现了一个分布式学习系统，该系统训练一个MNIST数字分类器。客户端在其本地数据上并行训练，服务器在每个通信轮次中平均客户端模型以创建全局模型。

### 程序要求
这个程序需要以下的Python模块：argparse, copy, torch, torchvision, pandas, numpy, matplotlib, tqdm 和 torch.multiprocessing。在运行程序之前，请确保你已经在你的Python环境中安装了这些模块。

### 程序描述

程序首先设置环境，并定义'device'变量，该变量将确定计算将在CPU上进行还是在可用的CUDA（GPU）上进行。
然后定义了'load_data()'函数，该函数将加载MNIST训练和测试数据集。
然后创建了'Classifier()'类，该类定义了学习过程中使用的神经网络模型的结构。
接下来是用于训练单个客户端('client_train()')，并行训练所有客户端('parallel_clients_train()')，平均客户端模型('avg_model()')，评估模型('evaluate()')和训练全局模型('global_train()')的函数。
在'main()'函数中，编排了整个学习过程。
最后，解析命令行参数以调整学习速率，客户端数量，周期数，批次大小，通信轮数以及选择的数据集。

### 如何运行

要运行程序，使用以下命令：

```
python main.py --lr 0.001 --num_clients 10 --epochs 5 --batch_size 32 --rounds 10 --dataset MNIST
```

你可以调整学习速率，客户端数量，周期数，批次大小，通信轮数以及数据集。

### 输出

该程序将输出一个图，显示每个通信轮次后全局模型的准确性。它还将这些数据以数据框的形式保存在一个前缀名为'global_results.xlsx'的Excel文件和一个名为'global_results.png'的png文件中。