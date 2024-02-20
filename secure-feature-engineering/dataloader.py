import torch
from torchvision import datasets, transforms
import syft as sy 

class Dataloader():
    def __init__(self):
        self.dataset = None
        self.loader = None

    def load_dataset(self, dataset, isNormalize=False, mean=0.1307, std=0.3081, isTrain=True):
        """
        Load dataset with normalizing
        """
        if dataset == "MNIST":
            if isNormalize:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((mean,), (std,))
                ])
                print("Normalizing...")
            else:
                transform = transforms.Compose([
                    transforms.ToTensor()
                ])
                print("Not normalizing...")

            self._mnist_loader(transform=transform, isTrain=isTrain)
        else:
            raise Exception("The dataset not found, MNIST is supported only")
            
        
    def _mnist_loader(self, transform, isTrain=True):
        """
        Load MINST dataset
        """
        # TODO: support more args change
        self.dataset = datasets.MNIST(root='./data', train=isTrain, download=True, transform=transform)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False)
