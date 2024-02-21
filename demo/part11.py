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

args = Arguments()  # create a new object from class Arguments
# print(args.batch_size)

# load train dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True)

# load test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True)

private_test_loader = []
for data, target in test_loader:
    private_test_loader.append((
        data.fix_precision().share(alice, bob, crypto_provider=crypto_provider, protocol="fss"),
        target.fix_precision().share(alice, bob, crypto_provider=crypto_provider, protocol="fss")
    ))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 500)  # doc: https://pytorch.org/docs/1.4.0/nn.html#linear
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        output = F.log_softmax(output, dim=1)   # doc: https://pytorch.org/docs/1.4.0/nn.functional.html
        loss = F.nll_loss(output, target)       # doc: https://pytorch.org/docs/1.4.0/nn.functional.html
        loss.backward()
        optimizer.step()    # update w
        if batch_idx % args.log_interval == 0:  # print once in every interval
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader) * args.batch_size,
                100. * batch_idx / len(train_loader), loss.item()))

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(1, args.epochs + 1):
    train(args, model, train_loader, optimizer, epoch)

def test(args, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
# test(args, model, test_loader)

model.fix_precision().share(alice, bob, crypto_provider=crypto_provider, protocol="fss")

def test(args, model, test_loader):
    model.eval()
    n_correct_priv = 0
    n_total = 0
    with torch.no_grad():
        for data, target in test_loader[:n_test_batches]:
            output = model(data)
            pred = output.argmax(dim=1) 
            n_correct_priv += pred.eq(target.view_as(pred)).sum()
            n_total += args.test_batch_size

            n_correct = n_correct_priv.copy().get().float_precision().long().item()
    
            print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
                n_correct, n_total,
                100. * n_correct / n_total))

test(args, model, private_test_loader)

