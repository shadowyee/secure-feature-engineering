import syft as sy
import torch

# participants
hook = sy.TorchHook(torch)
alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider") 


# create some variable and divide them into shares
x = torch.tensor([9])
y = torch.tensor([2])
# x_share = x.secret(secret_type="Additive", field=11).share(bob, alice, crypto_provider)
# y_share = y.secret(secret_type="Additive", field=11).share(bob, alice, crypto_provider)
x_share = x.share(alice, bob, crypto_provider=crypto_provider, protocol="fss")
y_share = y.share(alice, bob, crypto_provider=crypto_provider, protocol="fss")

import time 
start_time = time.time()
# division based on secret sharing
result_share = x_share < y_share

# get division result
result = result_share.get()
print("Time consumption: {:.4f}s".format(time.time() - start_time))

print("The result of secret division is:", result)