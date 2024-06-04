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

def secure_range(data):
    dim = len(data.shape)
    if dim > 1:
        raise AttributeError("Secure range Computation only support 1 dim data")
    
    # Generate secret shares
    shares = data.fix_precision(precision_fractional=prec).share(alice, bob, crypto_provider=crypto_provider)

    sort_vector = []    # Vector to record the local after sorting
    N = data.shape[0]

    sy.comm_total = 0   # Initial comm.
    start_time = time.time()  # Start timing

    # Calculate the range of data secretly between alice and bob
    for i in range(N):
        rank = shares[i] > shares[i]
        for j in range(N):
            if j == i:
                continue
            rank += shares[i] > shares[j]
        sort_vector.append(rank)

    max_bit_vector = [] # Bit vector to record the location of maximum
    min_bit_vector = [] # Bit vector to record the location of minimum

    max_idx = torch.Tensor([N-1]).fix_precision(precision_fractional=prec).share(alice, bob, crypto_provider=crypto_provider)    
    min_idx = torch.Tensor([0]).fix_precision(precision_fractional=prec).share(alice, bob, crypto_provider=crypto_provider)    
    
    # Initialize the value of maximum and minimum
    max_val = torch.Tensor([0]).fix_precision(precision_fractional=prec).share(alice, bob, crypto_provider=crypto_provider)    
    min_val = torch.Tensor([0]).fix_precision(precision_fractional=prec).share(alice, bob, crypto_provider=crypto_provider)    

    for i in range(N):
        max_bit = sort_vector[i] == max_idx
        max_bit_vector.append(max_bit)
        min_bit = sort_vector[i] == min_idx
        min_bit_vector.append(min_bit)
    
    for i in range(N):
        max_val += shares[i] * max_bit_vector[i]
        min_val += shares[i] * min_bit_vector[i]
    
    r = max_val - min_val

    end_time = time.time()  # End timing
    time_cost = end_time - start_time # Time cost
    communication_cost = sy.comm_total / (2 ** 20)  # Communication cost in MB

    range_result = float(r.get().child)/pow(10, prec) # Rconstruct the result

    print("The range result: {}".format(range_result))
    print("The communication cost is {:.3f} MB".format(communication_cost))
    print("The time cost is {:.3f} ms".format(time_cost * 1000))

# Input
data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
secure_range(data)