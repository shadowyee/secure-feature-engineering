# define the arguments
epochs = 10
n_test_batches = 200

# imports 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy

# create some virtualworker
hook = sy.TorchHook(torch) 
client = sy.VirtualWorker(hook, id="client")
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider") 

class Arguments():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 50
        self.epochs = epochs
        self.lr = 0.001
        self.log_interval = 100

args = Arguments()
print(args.batch_size)