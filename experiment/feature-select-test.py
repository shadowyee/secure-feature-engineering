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
# train_loader = dld.Dataloader()
# train_loader.load_dataset("MNIST", isTrain=True, batch_size=args.batch_size,isNormalize=True)      # Normalize the MNIST dataset

workers = parties.data_owners
crypto_provider = parties.crypto_provider


# Secure feature select from the whole dataset
def all_fs():
    data = torch.Tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [4, 5, 6], [4, 5, 6], [4, 5, 6]])
    target = torch.LongTensor([0, 3, 9, 4, 5, 7, 8])
    print("Data:", data)
    print("Label:", target)
    data_num = data.shape[0]
    # private_train_dataset = sfe.secret_share(train_loader.dataset.data[:10], workers, crypto_provider)
    # private_train_target = sfe.secret_share(sfe.one_hot_of(train_loader.dataset.targets[:10]), workers, crypto_provider)
    private_train_dataset = sfe.secret_share(data, workers, crypto_provider)
    private_train_target = sfe.secret_share(sfe.one_hot_of(target), workers, crypto_provider)
    alice, bob = parties.data_owners

    feature_num = 3
    # torch.Tensor.reshape(data, (-1, 784))
    
    data_f = private_train_dataset.view(-1, feature_num)         # Flatten the image of MNIST
    # print("Reshaped image size: {}".format(data_f.shape))                         # The first num refers to the batch size of dataset
    
    # data_f is a copy of data, the origin data shares would not be changed
    # So, here we have two different data shares: reshaped one and not reshaped one
    
    # Calculate the mean of every feature
    mean = torch.mean(private_train_dataset, dim=0)    
    # print(mean.get())

    mnist_classes = 10
    a = sfe.secret_share(torch.zeros(1), workers, crypto_provider)
    b = sfe.secret_share(torch.zeros(1), workers, crypto_provider)
    A = sfe.secret_share(torch.zeros(mnist_classes), workers, crypto_provider)
    B = sfe.secret_share(torch.zeros(mnist_classes), workers, crypto_provider)
    G_F = sfe.secret_share(torch.zeros(feature_num), workers, crypto_provider)
    dn = sfe.secret_share(torch.Tensor([data_num]), workers, crypto_provider)
    import time
    
    # data_num = train_loader.dataset.data.shape[0]
  
    for idx_f in range(0, feature_num):
        start_time = time.time()
        for i in range(0, data_num):
            flag_s = (mean[idx_f] < data_f[i, idx_f])
            b += flag_s
            for j in range(0, mnist_classes):
                
                flag_m = flag_s * private_train_target[i][j]     # The target is one-hot encoded
                B[j] += flag_m
                A[j] += private_train_target[i][j] - flag_m
        
        # a = args.batch_size - b
        a = dn - b
        for i in range(0, mnist_classes - 1):
            A[mnist_classes - 1] += A[i]
            B[mnist_classes - 1] += B[i]
        
        A[mnist_classes - 1] = a - A[mnist_classes - 1]
        B[mnist_classes - 1] = b - B[mnist_classes - 1]
        
        for i in range(0, mnist_classes - 1):
            if i == 0:
                mid_A = A[i] * A[i]
                mid_B = B[i] * B[i]
            else:
                mid_A += A[i] * A[i]
                mid_B += B[i] * B[i]
                
        G_le = a - mid_A / a
        G_gt = b - mid_B / b

        G_F[idx_f] = G_le + G_gt
        print("MS-GINI turn {}: {:.2f}s".format(idx_f, time.time() - start_time))
    
    G = G_F.get().child.child.child
    
    print(G)
    # torch.save(G, 'data/G_matrix_all.pt')

if __name__=="__main__":
    print("Feature Score:")
    all_fs()