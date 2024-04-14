# Refer to https://github.com/OpenMined/PySyft/blob/ryffel/0.2.x-fix-training/examples/tutorials
# Part 09 - Intro to Encrypted Programs.ipynb
# Step 3: SMPC Using PySyft

import torch
import syft as sy
import sys

hook = sy.TorchHook(torch)

alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")
copper = sy.VirtualWorker(hook, id="copper")

# secure addition via pysyft
x = torch.tensor([25]).share(bob,alice)
y = torch.tensor([5]).share(bob,alice)

def secure_add(x, y):
    print(type(x.child))        # check the type of x.child
    if isinstance(x.child, sy.AdditiveSharingTensor):
    # if isinstance(x, torch.Tensor):
        return x + y
    else:
        print("error")
        return 

# z1 = x + y
# z1 = secure_add(x, y)
print(sys.modules[__name__])
z1 = getattr(sys.modules[__name__], "secure_add")(x, y)
print ("secure addition:", z1.get())

# secure multiplication via pysyft
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")   # 利用可信第三方生成乘法三元组
x = torch.tensor([25]).share(bob,alice, crypto_provider=crypto_provider)
y = torch.tensor([5]).share(bob,alice, crypto_provider=crypto_provider)

print(type(x.child))
z2 = x * y
print ("secure multiplication:", z2.get())


# secure matrix multiplication via pysyft
x = torch.tensor([[1, 2],[3,4]]).share(bob,alice, crypto_provider=crypto_provider)
y = torch.tensor([[2, 0],[0,2]]).share(bob,alice, crypto_provider=crypto_provider)

print(type(x.child))
z3 = x.mm(y)  # matrix multiplication
ans3 = z3.get()
dim = len(ans3.shape)
print ("secure matrix multiplication:", ans3)

# three parties addition
x = torch.tensor([1]).share(alice, bob, copper, crypto_provider=crypto_provider)
y = torch.tensor([2]).share(alice, bob, copper, crypto_provider=crypto_provider)
z = torch.tensor([3]).share(alice, bob, copper, crypto_provider=crypto_provider)
sum = x + y + z
print("three parties addition:", sum.get())