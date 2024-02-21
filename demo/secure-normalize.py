import torch
from torchvision import datasets, transforms
import syft as sy 

hook = sy.TorchHook(torch)
# participants
names = ['alice', 'bob']
participants = []
for people in names:
    participants.append(sy.VirtualWorker(hook, id=people))

crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")


# Define transform to convert images to PyTorch tensors
transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))      # Don't normalize the pixel value
    ])

# Load the MNIST training dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

def secret_share(tensor, workers):
    """
    Transform to fixed precision and secret share a tensor
    """
    return (
        tensor
        .fix_precision()       
        .share(*workers, crypto_provider=crypto_provider, protocol="fss", requires_grad=True) # requires_grad is not important here
    )

def secure_normalize(data, mean, variance):
    """
    Normalize the data
    """
    assert torch.is_tensor(data)
    data -= mean
    data /= variance

for data, target in train_loader:
    secret_data = secret_share(data, names)
    secure_normalize(secret_data, 0.1307, 0.3081)

    shares = secret_data.get()
    print(shares[0])

    
exit()

# TODO: verify the accuracy of secure_normalize, comparing with the transforms.Normalize

# Nomalize test
train_dataset_test = datasets.MNIST(root='./data', train=True, download=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
train_loader_test = torch.utils.data.DataLoader(train_dataset_test, batch_size=len(train_dataset), shuffle=False)

cmp = None
for data, target in train_loader_test:
    cmp = data.fix_precision(precision_fractional=3)
    print(cmp[0])


