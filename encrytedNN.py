epochs = 1
# We don't use the whole dataset for efficiency purpose, but feel free to increase these numbers
# n_train_items = 320
# n_test_items = 320
n_train_items = 784 * 128
n_test_items = 784 * 128

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import time

class Arguments():
    def __init__(self):
        self.batch_size = 128
        self.test_batch_size = 128
        self.epochs = epochs
        self.lr = 0.02
        self.seed = 1
        self.log_interval = 1 # Log info at each batch
        self.precision_fractional = 3
        self.normalize_mean = 0.1307
        self.normalize_std = 0.3081

args = Arguments()

_ = torch.manual_seed(args.seed)

import syft as sy  # import the Pysyft library
import participants as pt

owners = ['alice', 'bob']
crypto_provider = 'crypto_provider'
parties = pt.Parties()
parties.init_parties(owners, crypto_provider)

def get_private_data_loaders(workers, crypto_provider):
    # import project modules
    import securefe as sfe
    import dataloader as dld

    train_loader = dld.Dataloader()
    train_loader.load_dataset("MNIST", isTrain=True, batch_size=args.batch_size)
    
    test_loader = dld.Dataloader()
    test_loader.load_dataset("MNIST", isTrain=False, batch_size=args.test_batch_size)
    
    mean = args.normalize_mean
    std = args.normalize_std

    private_train_loader = [
        (sfe.secure_normalize(sfe.secret_share(data, workers, crypto_provider), mean, std), 
         sfe.secret_share(sfe.one_hot_of(target), workers, crypto_provider)) 
        for i, (data, target) in enumerate(train_loader.loader)
        if i < n_train_items / args.batch_size
    ]

    private_test_loader = [
        (sfe.secure_normalize(sfe.secret_share(data, workers, crypto_provider), mean, std), 
         sfe.secret_share(target.float(), workers, crypto_provider))
        for i, (data, target) in enumerate(test_loader.loader)
        if i < n_test_items / args.test_batch_size
    ]

    return private_train_loader, private_test_loader
    
    
private_train_loader, private_test_loader = get_private_data_loaders(
    workers=parties.data_owners,
    crypto_provider=parties.crypto_provider
)

# print(private_test_loader)

# exit()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # x = x.view(-1, 784)           view() is not working here, use torch.Tensor.reshape() instead
        x = torch.Tensor.reshape(x, (-1, 784))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

def train(args, model, private_train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(private_train_loader): # <-- now it is a private dataset
        start_time = time.time()
        
        optimizer.zero_grad()

        output = model(data)
        # loss = F.nll_loss(output, target)  <-- not possible here
        batch_size = output.shape[0]
    
        loss = ((output - target)**2).sum().refresh()/batch_size
        
        loss.backward()
        # print(loss.get())
        # exit()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            loss = loss.get().float_precision()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s'.format(
                epoch, batch_idx * args.batch_size, len(private_train_loader) * args.batch_size,
                100. * batch_idx / len(private_train_loader), loss.item(), time.time() - start_time))
            
def test(args, model, private_test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in private_test_loader:
            start_time = time.time()
            
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()

    correct = correct.get().float_precision()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct.item(), len(private_test_loader)* args.test_batch_size,
        100. * correct.item() / (len(private_test_loader) * args.test_batch_size)))

model = Net()
model = model.fix_precision().share(*parties.data_owners, crypto_provider=parties.crypto_provider, protocol="fss", requires_grad=True)

optimizer = optim.SGD(model.parameters(), lr=args.lr)
optimizer = optimizer.fix_precision() 

model.train()
for epoch in range(1, args.epochs + 1):
    train(args, model, private_train_loader, optimizer, epoch)
    test(args, model, private_test_loader)


