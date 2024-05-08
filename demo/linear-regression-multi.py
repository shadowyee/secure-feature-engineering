import torch
import syft as sy
import time
import csv

hook = sy.TorchHook(torch)

alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

def linear_regression_boston_dataset():
    from sklearn.datasets import load_boston

    # 加载波士顿房价数据集
    boston = load_boston()

    # 提取特征和目标值
    X, y = boston.data, boston.target
    N = X.shape[0]
    M = X.shape[1]
    prec = 3

    # 特征标准化
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    
    X = torch.tensor(X, dtype=torch.float32).fix_precision(precision_fractional=prec).share(alice, bob, crypto_provider=crypto_provider)
    y = torch.tensor(y, dtype=torch.float32).fix_precision(precision_fractional=prec).share(alice, bob, crypto_provider=crypto_provider)

    theta = torch.zeros(M).fix_precision().share(alice, bob, crypto_provider=crypto_provider)

    lr = 0.1

    time_idx = []
    cost_idx = []

    num_iterations = 100

    start_time = time.time()
    for i in range(num_iterations):
        h = X.mm(theta)
        # E = (h - y).view(-1, 1)
        E = torch.Tensor.reshape((h-y), (-1, 1))
        cost = E.transpose(0, 1).mm(E) / (2 * N)

        cidx = float(cost.get()) / pow(10, prec)
        cost_idx.append(cidx)
        time_idx.append("{:.6f}".format(time.time() - start_time))
        print(cidx)
        mid = X.transpose(0,1).mm(E).view(-1)
        theta = theta - mid / (N / lr)

    print(theta.get())
    with open('../experiment/linear-multi-pysyft.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time', 'Cost'])
        writer.writerows(zip(time_idx, cost_idx))  
        
if __name__ == "__main__":
    linear_regression_boston_dataset()