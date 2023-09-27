import argparse
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.multiprocessing as mp
from data import load_data
from model import Classifier
from train import global_train, avg_model
from evaluate import evaluate

# Set the start method for multiple processes
mp.set_start_method('spawn', force=True)

# Check if CUDA is available, else use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
