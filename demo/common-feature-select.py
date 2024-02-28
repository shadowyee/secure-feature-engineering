import torch
import sys
sys.path.append('..')
import dataloader as dld
import securefe as sfe
import syft as sy
from torchvision import datasets, transforms

class Arguments():
    def __init__(self):
        self.batch_size = 128
        self.feature_num = 784
        self.classes = 10
args = Arguments()

train_loader = dld.Dataloader()
train_loader.load_dataset("MNIST", isTrain=True, batch_size=args.batch_size, isNormalize=False)      # Normalize the MNIST dataset
data_fla = train_loader.dataset.data.float().view(-1, args.feature_num)

def _cal_Gmatrix():
    feature_mean = torch.mean(data_fla, dim=0)

    
    G = torch.zeros(args.feature_num)

    sample_num = data_fla.shape[0]
    fnum = args.feature_num
    fmean = feature_mean
    classes = args.classes
    target = sfe.one_hot_of(train_loader.dataset.targets)

    import time
    for idx_f in range(0, fnum):
        start_time = time.time()
        a = torch.zeros(1)
        b = torch.zeros(1)
        A = torch.zeros(args.classes)
        B = torch.zeros(args.classes)

        for i in range(0, sample_num):
            flag_s = (fmean[idx_f] < data_fla[i, idx_f])
            b += flag_s
            for j in range(0, classes):
                flag_m = flag_s * target[i][j]     # The target is one-hot encoded
                B[j] += flag_m
                A[j] += target[i][j] - flag_m
        
        a = sample_num - b
        for i in range(0, classes - 1): 
            A[classes - 1] += A[i]
            B[classes - 1] += B[i]
        A[classes - 1] = a - A[classes - 1]
        B[classes - 1] = b - B[classes - 1]
        G_le = a - (torch.Tensor.dot(A, A)) / a if not a == 0 else 0
        G_gt = b - (torch.Tensor.dot(B, B)) / b if not b == 0 else 0
       
        G[idx_f] = G_le + G_gt
        print("MS-GINI turn {}: {:.2f}s".format(idx_f, time.time() - start_time))
        print(G[idx_f])
    print(G)
    import os
    if not os.path.exists('result'):
    # Create the directory if it doesn't exist
        os.makedirs('result')
    torch.save(G, 'result/common_G.pt')
    return G

def _read_G():
    G = torch.load('result/common_G.pt')
    print("G shape: ".format(G.shape))
    return G

def _select_feature(G, select_num):
    fnum = args.feature_num
    Gmax = G.max()
    I = torch.zeros(fnum)
    T = torch.zeros(fnum, select_num)
    
    import time
    for i in range(0, select_num):
        start_time = time.time()
        I[i] = torch.min(G)
        for j in range(0, fnum):
            flag_k = I[i] == j
            T[j][i] = flag_k
            G[j] += flag_k * (Gmax - G[j])
        print("Feature Select turn {}: {:.2f}s".format(i, time.time() - start_time))
    D = data_fla.mm(T)
    print("D shape: ".format(D.shape))
    dataset = {
        'data': D,
        'labels': train_loader.dataset.targets
    } 

    import os
    if not os.path.exists('result/MNIST_{}'.format(select_num)):
    # Create the directory if it doesn't exist
        os.makedirs('result/MNIST_{}'.format(select_num))
    torch.save(dataset, 'result/MNIST_{}/data_fnum{}.pt'.format(select_num, select_num))

if __name__=="__main__":
    # _cal_Gmatrix()
    G = _read_G()
    for i in range(1, 8):
        select_num = i * 100
        _select_feature(G, select_num)
