# import third-party package
import torch
from torchvision import datasets, transforms
import syft as sy 

def secret_share(data: torch.Tensor, workers, crypto_provider):
    """
    Transform to fixed precision and secret share a tensor
    """
    return (
        data
        .fix_precision()       
        .share(*workers, crypto_provider=crypto_provider, protocol="fss", requires_grad=True)
    )

def secure_normalize(secret_data: torch.Tensor, mean, variance):
    """
    Normalize the secret shares of data
    """
    # input = (input - mean)/variance
    secret_data -= mean
    secret_data /= variance
    
    return secret_data 

def one_hot_of(index_tensor):
    """
    Transform to one hot tensor
    
    Example:
        [0, 3, 9]
        =>
        [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    """
    onehot_tensor = torch.zeros(*index_tensor.shape, 10) # 10 classes for MNIST
    onehot_tensor = onehot_tensor.scatter(1, index_tensor.view(-1, 1), 1)
    return onehot_tensor

def secure_Gmatrix(data: torch.Tensor, target: torch.Tensor, fnum, classes, workers, crypto_provider):
    """
    Calculate GINI impurity matrix of all features 
    """
    # data_fla = torch.Tensor.reshape(data, (-1, fnum))
    data_fla = data.view(-1, fnum)
    # data_fla is a copy of data, the origin data shares would not be changed
    # So, here we have two different data shares: reshaped one and not reshaped one

    fmean = torch.mean(data_fla, dim=0)

    # Initialize the necessary vectors
    a = secret_share(torch.zeros(1), workers, crypto_provider)
    b = secret_share(torch.zeros(1), workers, crypto_provider)
    A = secret_share(torch.zeros(classes), workers, crypto_provider)
    B = secret_share(torch.zeros(classes), workers, crypto_provider)
    G = secret_share(torch.zeros(classes), workers, crypto_provider)

    sample_num = data_fla.shape[0]

    import time

    for idx_f in range(0, fnum):
        start_time = time.time()
        # TODO: fix: a, b, A, B should be initialized here
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
        
        # TODO: b can not be zero
        G_le = a - (torch.Tensor.dot(A, A)) / a
        G_gt = b - (torch.Tensor.dot(B, B)) / b

        G[idx_f] = G_le + G_gt
        print("MS-GINI turn {}: {:.2f}s".format(idx_f, time.time() - start_time))
    
    return G

def secure_fselect(data: torch.Tensor, G, fnum, snum, workers, crypto_provider):
    """
    Retain the features base on the GINI impurity
    """
    data_fla = data.view(-1, fnum)
    max_num = G.max(dim=0)

    # Initialize the necessary matrix
    I = secret_share(torch.zeros(fnum), workers, crypto_provider)
    T = secret_share(torch.zeros(fnum, snum), workers, crypto_provider)

    import time

    for i in range(0, snum):
            start_time = time.time()
            I[i] = G.min(dim=0)
            for j in range(0, fnum):
                flag_k = I[i] == j
                T[j][i] = flag_k
                G[j] += flag_k * (max_num - G[j])
            print("Feature Select turn {}: {:.2f}s".format(i, time.time() - start_time))

    return data_fla.mm(T)
    

# TODO: implement other secure feature engineering method