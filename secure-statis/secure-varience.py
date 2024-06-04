import torch
import syft as sy
import time

# Initialize hook
hook = sy.TorchHook(torch)

# Create parties alice, bob and multiplication triple provider
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

# precision of fix point num
prec = 4

def secure_varience(data):
    dim = len(data.shape)
    if dim > 1:
        raise AttributeError("Secure varience Computation only support 1 dim data")

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

    end_time = time.time()  # End timing
    time_cost = end_time - start_time # Time cost
    communication_cost = sy.comm_total / (2 ** 20)  # Communication cost in MB

    vari_result = float(vari.get().child)/pow(10, prec) # Rconstruct the result

    print("The varience result: {}".format(vari_result))
    print("The communication cost is {:.3f} MB".format(communication_cost))
    print("The time cost is {:.3f} ms".format(time_cost * 1000))

# Input
data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
secure_varience(data)