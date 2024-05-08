import torch
import numpy as np
import time
import csv

def common_dataset():
    X = torch.Tensor([[1, 2], [2, 3], [3, 4]])
    # y = torch.Tensor([[5], [7], [9]])
    y = torch.Tensor([5, 7, 9])
    theta = torch.zeros(X.shape[1])
    
    lr = 0.01

    num_iterations = 500
    for i in range(num_iterations):
        # print("X", X.shape)
        # print("theta", theta.shape)
        h = torch.matmul(X, theta)
        # print("h", h.shape)
        # print("y", y.shape)
        E = h - y

        # print("E", E.shape)
        print((0.5 / X.shape[0]) * torch.matmul(E.t(), E))
        
        mid = torch.matmul(X.t(), E)
        theta = theta - (lr / X.shape[0]) * mid
    
    print(theta)
    # gradient_descent(X, Y, w_start, b_start, 0.01, 2000, N)

def boston_dataset():
    from sklearn.datasets import load_boston
    from sklearn.preprocessing import StandardScaler

    # 加载波士顿房价数据集
    boston = load_boston()

    # 提取特征和目标值
    X, y = boston.data, boston.target

    # 数据预处理，标准化特征
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # 特征标准化
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    # 将数据转换为PyTorch张量
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    theta = torch.zeros(X.shape[1])
    
    lr = 0.1

    num_iterations = 100

    time_idx = []
    cost_idx = []

    start_time = time.time()
    for i in range(num_iterations):
        h = torch.matmul(X, theta)
        E = h - y
        cost = (0.5 / X.shape[0]) * torch.matmul(E.t(), E)
        cost_idx.append(float(cost))
        time_idx.append("{:.6f}".format(time.time() - start_time))
        print(cost)
        mid = torch.matmul(X.t(), E)
        theta = theta - (lr / X.shape[0]) * mid
    
    print(theta)

    with open('../experiment/linear-multi-common.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time', 'Cost'])
        writer.writerows(zip(time_idx, cost_idx)) 

if __name__ == "__main__":
    # common_dataset()
    boston_dataset()