import torch
from torchvision import datasets, transforms
import numpy as np
from scipy import stats

# Define transform to convert images to PyTorch tensors
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))      # Normalize the pixel value
    ])

# Load the MNIST training dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Create a data loader for the training dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

# Calculate the mean of the dataset
mean = 0.0
median = 0.0
mode = 0.0
variance = 0.0
total_images = 0

# Concat all images into a single tensor
# all_images = torch.cat([image for image, _ in train_dataset], dim=0)     
# mean = torch.mean(all_images)
# median = torch.median(all_images)
# mode = stats.mode(all_images.numpy().flatten()).mode[0]  # Use scipy.stats.mode for mode calculation
# variance = torch.var(all_images)

for i, (images, _) in enumerate(train_loader):
    batch_size = images.size(0)
    images = images.view(batch_size, -1)  # Flatten the images
    # mean += torch.mean(images, dim=0) * batch_size
    mean = torch.mean(images, dim=0) * batch_size       # The DataLoader has only once iteration, so use the assignment instead
    median = torch.median(images, dim=0)
    mode = stats.mode(images.numpy().flatten()).mode[0]
    variance = torch.var(images)
    total_images += batch_size
    # print(i)        # The DataLoader object has only once iteration

mean /= total_images

print("Mean of the MNIST dataset:", mean)
print("Median of the MNIST dataset:", median)
print("Mode of the MNIST dataset:", mode)
print("Variance of the MNIST dataset:", variance)

