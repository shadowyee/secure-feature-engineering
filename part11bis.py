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
print("dataset size: " + str(dataset_size))
print("class name: " + str(class_names))

model = models.resnet18(pretrained=True)
# Here the size of each output sample is set to 2.
model.fc = nn.Linear(model.fc.in_features, 2)
state = torch.load("./model/resnet18_ants_bees.pt", map_location='cpu')
model.load_state_dict(state)
model.eval()
# This is a small trick because these two consecutive operations can be switched without
# changing the result but it reduces the number of comparisons we have to compute
model.maxpool, model.relu = model.relu, model.maxpool

import syft as sy

hook = sy.TorchHook(torch) 
data_owner = sy.VirtualWorker(hook, id="data_owner")
model_owner = sy.VirtualWorker(hook, id="model_owner")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

# Remove compression to have faster communication, because compression time 
# is non-negligible: we send to workers crypto material which is very heavy
# and pseudo-random, so compressing it takes a long time and isn't useful:
# randomness can't be compressed, otherwise it wouldn't be random!
from syft.serde.compression import NO_COMPRESSION
sy.serde.compression.default_compress_scheme = NO_COMPRESSION

data, true_labels = next(iter(dataloader))
print("data: " + str(data))
data_ptr = data.send(data_owner)

# We store the true output of the model for comparison purpose
true_prediction = model(data)
model_ptr = model.send(model_owner)

print(data_ptr)

encryption_kwargs = dict(
    workers=(data_owner, model_owner), # the workers holding shares of the secret-shared encrypted data
    crypto_provider=crypto_provider, # a third party providing some cryptography primitives
    protocol="fss", # the name of the crypto protocol, fss stands for "Function Secret Sharing"
    precision_fractional=4, # the encoding fixed precision (i.e. floats are truncated to the 4th decimal)
)

encrypted_data = data_ptr.encrypt(**encryption_kwargs).get()
encrypted_model = model_ptr.encrypt(**encryption_kwargs).get()

start_time = time.time()

encrypted_prediction = encrypted_model(encrypted_data)
encrypted_labels = encrypted_prediction.argmax(dim=1)

print(time.time() - start_time, "seconds")

labels = encrypted_labels.decrypt()

print("Predicted labels:", labels)
print("     True labels:", true_labels)

print(encrypted_prediction.decrypt())
print(true_prediction)
