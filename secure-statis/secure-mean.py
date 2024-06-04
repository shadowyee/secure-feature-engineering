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

def secure_mean(data):
    dim = len(data.shape)
    if dim > 2:
        raise AttributeError("Secure mean Computation only support less than 3 dims")

    # Generate secret shares
    shares = data.fix_precision(precision_fractional=prec).share(bob,alice, crypto_provider=crypto_provider)
    
    sy.comm_total = 0   # Initial comm.
    start_time = time.time()  # Start timing

    # Calculate the mean of data secretly between alice and bob
    mean = []
    if dim == 1:
        N = data.shape[0]
        sum = shares.sum()
        mean.append(sum * (1/N))
    elif dim == 2:
        # Calculate the mean of each row of a matrix
        N = data.shape[1]
        R = data.shape[0]
        for i in range(R):
            s = shares[i].sum()
            m = s * (1/N)
            mean.append(m)

    end_time = time.time()  # End timing
    time_cost = end_time - start_time # Time cost
    communication_cost = sy.comm_total / (2 ** 20)  # Communication cost in MB

    mean_result = []
    for m in mean:
        mean_result.append(float(m.get().child)/pow(10, prec)) # Rconstruct the result

    print("The mean result: {}".format(mean_result))
    print("The communication cost is {:.3f} MB".format(communication_cost))
    print("The time cost is {:.3f} ms".format(time_cost * 1000))

# Input
data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# data = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
secure_mean(data)
