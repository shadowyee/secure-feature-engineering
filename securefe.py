# import third-party package
import torch
from torchvision import datasets, transforms
import syft as sy 

def secret_share(data, workers, crypto_provider):
    """
    Transform to fixed precision and secret share a tensor
    """
    assert torch.is_tensor(data)
    return (
        data
        .fix_precision()       
        .share(*workers, crypto_provider=crypto_provider, protocol="fss", requires_grad=True)
    )

def secure_normalize(secret_data, mean, variance):
    """
    Normalize the secret shares of data
    """
    assert torch.is_tensor(secret_data)
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

# TODO: implement other secure feature engineering method