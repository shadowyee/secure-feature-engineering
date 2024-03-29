import torch
from torchvision import datasets, transforms
import syft as sy 

class Dataloader():
    def __init__(self):
        self.dataset = None
        self.loader = None

    def load_dataset(self, dataset, batch_size, isNormalize=False, mean=0.1307, std=0.3081, isTrain=True):
        """
        Load dataset with normalizing
        """
        if dataset == "MNIST":
            if isNormalize:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((mean,), (std,))
                ])
                print("Load dataset: load {} and normalize...".format(dataset))
            else:
                transform = transforms.Compose([
                    transforms.ToTensor()
                ])
                print("Load dataset: load {} and not normalize...".format(dataset))

            self._mnist_loader(transform=transform, isTrain=isTrain, batch_size=batch_size)
        elif dataset == "MNIST_100":
            if isNormalize:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((mean,), (std,))
                ])
                print("Load dataset: load {} and normalize...".format(dataset))
            else:
                transform = transforms.Compose([
                    transforms.ToTensor()
                ])
                print("Load dataset: load {} and not normalize...".format(dataset))
            
            self._processed_mnist_loader(transform=transform, isTrain=isTrain, batch_size=batch_size)
        else:
            raise Exception("The dataset not found, MNIST is supported only")
            
        
    def _mnist_loader(self, transform, batch_size, isTrain=True):
        """
        Load MINST dataset
        """
        # TODO: support more args change
        self.dataset = datasets.MNIST(root='./data', train=isTrain, download=True, transform=transform)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

    def _processed_mnist_loader(self, transform, batch_size, isTrain=True):
        dataset = torch.load('./demo/result/MNIST_100/data_fnum100.pt')
        data = dataset['data']
        labels = dataset['labels']
        self.dataset = torch.utils.data.TensorDataset(data, labels)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=False)