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
train_loader.load_dataset("MNIST", isTrain=True, batch_size=args.batch_size,isNormalize=True)      # Normalize the MNIST dataset

workers = parties.data_owners
crypto_provider = parties.crypto_provider


# Secure feature select from the whole dataset
def all_fs():
    private_train_dataset = sfe.secret_share(train_loader.dataset.data, workers, crypto_provider)
    private_train_target = sfe.secret_share(sfe.one_hot_of(train_loader.dataset.targets), workers, crypto_provider)
    alice, bob = parties.data_owners

    feature_num = 784
    # torch.Tensor.reshape(data, (-1, 784))
    data_f = private_train_dataset.view(-1, feature_num)         # Flatten the image of MNIST
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
    
    import time
    
    data_num = train_loader.dataset.data.shape[0]
    for idx_f in range(0, feature_num):
        start_time = time.time()
        for i in range(0, data_num):
            flag_s = (mean[idx_f] < data_f[i, idx_f])
            b += flag_s
            for j in range(0, mnist_classes):
                
                flag_m = flag_s * private_train_target[i][j]     # The target is one-hot encoded
                B[j] += flag_m
                A[j] += private_train_target[i][j] - flag_m
        
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
    
    G = G_F.get().child.child.child
    torch.save(G, 'data/G_matrix_all.pt')

# Secure feature select from a batch
def batch_fs():
    private_train_loader = [
        (sfe.secret_share(data,  workers, crypto_provider), sfe.secret_share(sfe.one_hot_of(target), workers, crypto_provider))
        for i, (data, target) in enumerate(train_loader.loader)
        if i < 10
    ]

    for idx, (data, target) in enumerate(private_train_loader):
        # print("{}: {}".format(idx, data))
        alice, bob = parties.data_owners

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
        
        G = G_F.get().child.child.child
        torch.save(G, 'data/G_matrix.pt')
        break

if __name__=="__main__":
    all_fs()