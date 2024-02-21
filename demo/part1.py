# import necessary package
import sys

import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F

import syft as sy

torch.tensor([1,2,3,4,5])
print("Torch version: " + torch.__version__)        # display the torch version, click https://pytorch.org/docs/1.4.0/ for detailed doc

# execute a simple maching learning example
x = torch.tensor([1,2,3,4,5])   # torch.Tensor is an alias for the default tensor type (torch.FloatTensor).
y = x + x
print("The result of the simple example: " + str(y))

# send message to another machine
hook = sy.TorchHook(torch)      
bob = sy.VirtualWorker(hook, id="bob")  # create a "pretend" person--bob

x = torch.tensor([1,2,3,4,5])
y = torch.tensor([1,1,1,1,1])

x_ptr = x.send(bob)     # send message working with POINTERs
y_ptr = y.send(bob)

print("Pointer x_ptr: " + str(x_ptr))   # POINTER is actually created by Wrapper

print("Machine bob now have two tensors: " + str(bob._objects))  # _objects is a private attributes
z = x_ptr + x_ptr       # add the wrapper up and send it to the bob
print("Adding the POINTERs: " + str(z))      # display the adding result
print("Sending the addition result to bob: " + str(bob._objects))     # bob has new attribute now 

# POINTER do not hold data themselves but contain metadata tensor about a tensor
print("========================================")
print(x_ptr)
print("Some attributes of pointers: ")
print(x_ptr.location)           # the location POINTER pointing to
print(x_ptr.id_at_location)     # the id of tensor stored at the location
print(x_ptr.id)                 # the id of POINTER
print(x_ptr.owner)              # the owner of the POINTER
print("========================================")

# me is created when calling TorchHook() as a VitualWorker
me = sy.local_worker
print("Me is created automatically by sy.TorchHook(): " + str(me))
print("========================================")

# get back the Tensor
print("Before getting back the tensor: ")
print(str(bob._objects))
print(x_ptr)
print(y_ptr)
print(z)
print("Getting back the tensor: " + str(x_ptr.get()) + "," + str(y_ptr.get()) + "," + str(z.get()))
print("After getting back the tensor: " + str(bob._objects))
print("========================================")

# Send, Compute and Get
print("Send the tensors to the bob and perform the computation: ")
x = torch.tensor([1, 2, 3, 4, 5]).send(bob)
y = torch.tensor([1, 1, 1, 1, 1]).send(bob)
z = x + y   # add the tensor up in the bob machine
# z = torch.add(x, y)   # perform the computation by the Torch's operation
print(z) 
print(bob._objects)     # now bob has three tensors, including x, y and x + y
z.get()
print("Get the result from the bob: " + str(bob._objects))
print("========================================")

# backpropagation
x = torch.tensor([1, 2, 3, 4, 5.], requires_grad=True).send(bob)
y = torch.tensor([1, 1, 1, 1, 1.], requires_grad=True).send(bob)
z = (x + y).sum()   # compute in the remote machine
z.backward()        # compute the gradients
x = x.get()         # get back x
print(x.grad)       # gradients are accumulated  into .grad attribute
y = y.get()         # get back y
print(y.grad)

u = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
out = u.sum()       # compute in the local machine
out.backward()      # compute the gradients
print(u.grad)       # gradients are accumulated  into .grad attribute
