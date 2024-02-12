import torch
torch.set_num_threads(1) # We ask torch to use a single thread as we run async code which conflicts with multithreading
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = 'data/hymenoptera_data'
image_dataset = datasets.ImageFolder(data_dir + '/val', data_transform)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=2, shuffle=True, num_workers=4)

dataset_size = len(image_dataset)
class_names = image_dataset.classes

model = models.resnet18(pretrained=True)
# Here the size of each output sample is set to 2.
model.fc = nn.Linear(model.fc.in_features, 2)
state = torch.load("./model/resnet18_ants_bees.pt", map_location='cpu')
model.load_state_dict(state)
model.eval()
# This is a small trick because these two consecutive operations can be switched without
# changing the result but it reduces the number of comparisons we have to compute
model.maxpool, model.relu = model.relu, model.maxpool

