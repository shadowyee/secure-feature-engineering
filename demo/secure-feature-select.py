import torch
import sys
sys.path.append('..')
import securefe as sfe
import dataloader as dld

import syft as sy  

class Arguments():
    def __init__(self):
        self.batch_size = 128

args = Arguments()

# Initialize the parties
import participants as pt

owner_names = ['alice', 'bob']
provider_name = 'crypto_provider'
parties = pt.Parties()
parties.init_parties(owner_names, provider_name)

# Load MNIST dataset
train_loader = dld.Dataloader()
train_loader.load_dataset("MNIST", isTrain=True, batch_size=args.batch_size, isNormalize=True)      # Normalize the MNIST dataset

workers = parties.data_owners
crypto_provider = parties.crypto_provider

private_train_loader = [
    (sfe.secret_share(data,  workers, crypto_provider), sfe.secret_share(sfe.one_hot_of(target), workers, crypto_provider))
    for i, (data, target) in enumerate(train_loader.loader)
    if i < 10
]

for idx, (data, target) in enumerate(private_train_loader):
    feature_num = 784
    
    # torch.Tensor.reshape(data, (-1, 784))
    data_f = data.view(-1, feature_num)         # Flatten the image of MNIST
    print("Reshaped image size: {}".format(data_f.shape))                         # The first num refers to the batch size of dataset
    # data_f is a copy of data, the origin data shares would not be changed
    # So, here we have two different data shares: reshaped one and not reshaped one
    
    # Calculate the mean of every feature
    mean = torch.mean(data_f, dim=0)    
    # print(mean.get())

    mnist_classes = 10
    a = sfe.secret_share(torch.zeros(1), workers, crypto_provider)
    b = sfe.secret_share(torch.zeros(1), workers, crypto_provider)
    A = sfe.secret_share(torch.zeros(mnist_classes), workers, crypto_provider)
    B = sfe.secret_share(torch.zeros(mnist_classes), workers, crypto_provider)
    G_F = sfe.secret_share(torch.zeros(feature_num), workers, crypto_provider)

    # Test code TODO: Remember to delete them
    # mean = mean.get()
    # data_f = data_f.get()

    # for idx_f in range(0, feature_num):
    #     for i in range(0, args.batch_size):
    #         print("data_f: {} mean: {}".format(data_f[i, idx_f], mean[idx_f]))
    
    import time
    
    for idx_f in range(0, feature_num):
        start_time = time.time()
        for i in range(0, args.batch_size):
            flag_s = (mean[idx_f] < data_f[i, idx_f])
            b += flag_s
            for j in range(0, mnist_classes):
                
                flag_m = flag_s * target[i][j]     # The target is one-hot encoded
                B[j] += flag_m
                A[j] += target[i][j] - flag_m
        
        a = args.batch_size - b

        for i in range(0, mnist_classes - 1):
            A[mnist_classes - 1] += A[i]
            B[mnist_classes - 1] += B[i]
        
        A[mnist_classes - 1] = a - A[mnist_classes - 1]
        B[mnist_classes - 1] = b - B[mnist_classes - 1]

        G_le = a - (torch.Tensor.dot(A, A)) / a
        G_gt = b - (torch.Tensor.dot(B, B)) / b

        G_F[idx_f] = G_le + G_gt
        print("MS-GINI turn {}: {:.2f}s".format(idx_f, time.time() - start_time))
    
    # max_float32 = torch.finfo(torch.float32).max
    max_float32 = G_F.max(dim=0)

    for i in range(1, 7): 
        selected_fnum = i * 100
        I = sfe.secret_share(torch.zeros(feature_num), workers, crypto_provider)
        T = sfe.secret_share(torch.zeros(feature_num, selected_fnum), workers, crypto_provider)

        for i in range(0, selected_fnum):
            start_time = time.time()
            I[i] = G_F.min(dim=0)
            for j in range(0, feature_num):
                flag_k = I[i] == j
                T[j][i] = flag_k
                G_F[j] += flag_k * (max_float32 - G_F[j])
            print("Feature Select turn {}: {:.2f}s".format(i, time.time() - start_time))

        D = data_f.mm(T)
        D_get = D.get().child.child.child

        torch.save(D_get, 'data/data_fnum{}.pt'.format(selected_fnum))

    break