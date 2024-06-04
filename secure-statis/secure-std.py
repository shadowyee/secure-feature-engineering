import torch
import syft as sy
import time
import math

# Initialize hook
hook = sy.TorchHook(torch)

# Create parties alice, bob and multiplication triple provider
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

# precision of fix point num
prec = 4

def secure_std(data):
    dim = len(data.shape)
    if dim > 1:
        raise AttributeError("Secure std Computation only support 1 dim data")

    # Generate secret shares
    shares = data.fix_precision(precision_fractional=prec).share(bob,alice, crypto_provider=crypto_provider)

    sy.comm_total = 0   # Initial comm.
    start_time = time.time()  # Start timing

    # Calculate the mean of data secretly between alice and bob
    N = data.shape[0]
    sum = shares.sum()
    mean = sum * (1/N)

    # Calculate the varience of data secretly between alice and bob
    for i in range(N):
        diff = shares[i] - mean
        if i == 0:
            vari = diff * diff
        else:
            vari += diff * diff
    vari = vari * (1/N)

    v = vari.get() # Reconstruct the varience result
    u = math.exp(-2.2*(v/2 + 0.2)) + 0.198046875 # Calculate the exp locally
    u = torch.tensor(u)

    v_sh = v.share(alice, bob, crypto_provider=crypto_provider)
    u_sh = u.fix_precision(precision_fractional=prec).share(alice, bob, crypto_provider=crypto_provider)
    
    # Calculate the std of data secretly between alice and bob
    iter_num = 6 # The iteration times of Newton's method
    for _ in range(iter_num):
        u_sh = u_sh * (1.5 - v_sh * (1/2) * u_sh * u_sh)
    
    std = v_sh * u_sh

    end_time = time.time()  # End timing
    time_cost = end_time - start_time # Time cost
    communication_cost = sy.comm_total / (2 ** 20)  # Communication cost in MB

    std_result = float(std.get().child)/pow(10, prec) # Rconstruct the result

    print("The std result: {}".format(std_result))
    print("The communication cost is {:.3f} MB".format(communication_cost))
    print("The time cost is {:.3f} ms".format(time_cost * 1000))

# Input
data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
secure_std(data)