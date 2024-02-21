# Refer to https://github.com/OpenMined/PySyft/blob/ryffel/0.2.x-fix-training/examples/tutorials
# Part 09 - Intro to Encrypted Programs.ipynb
# Step 3: SMPC Using PySyft

import torch
import syft as sy
hook = sy.TorchHook(torch)

alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")

# secure addition via pysyft
x = torch.tensor([25]).share(bob,alice)
y = torch.tensor([5]).share(bob,alice)
z1 = x + y
print ("secure addition:", z1.get())

# secure multiplication via pysyft
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")   # 利用可信第三方生成乘法三元组
x = torch.tensor([25]).share(bob,alice, crypto_provider=crypto_provider)
y = torch.tensor([5]).share(bob,alice, crypto_provider=crypto_provider)

z2 = x * y
print ("secure multiplication:", z2.get())


# secure matrix multiplication via pysyft
x = torch.tensor([[1, 2],[3,4]]).share(bob,alice, crypto_provider=crypto_provider)
y = torch.tensor([[2, 0],[0,2]]).share(bob,alice, crypto_provider=crypto_provider)

z3 = x.mm(y)  # matrix multiplication
print ("secure matrix multiplication:", z3.get())
