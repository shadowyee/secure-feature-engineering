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

def secure_median(data):
    dim = len(data.shape)
    if dim > 1:
        raise AttributeError("Secure median Computation only support 1 dim data")

    # Generate secret shares
    shares = data.fix_precision(precision_fractional=prec).share(alice, bob, crypto_provider=crypto_provider)
    
    sort_vector = []    # Vector to record the local after sorting
    N = data.shape[0]
    
    sy.comm_total = 0   # Initial comm.
    start_time = time.time()  # Start timing

    # Calculate the median of data secretly between alice and bob
    for i in range(N):
        rank = shares[i] > shares[i]
        for j in range(N):
            if j == i:
                continue
            rank += shares[i] > shares[j]
        sort_vector.append(rank)

    bit_vector = [] # Bit vector to record the location of median
    mid_idx = torch.Tensor([N // 2]).fix_precision(precision_fractional=prec).share(alice, bob, crypto_provider=crypto_provider)    
    median = torch.tensor([0]).fix_precision(precision_fractional=prec).share(alice, bob, crypto_provider=crypto_provider)  # Initialize the median
    
    if N % 2 == 1:  # When the number of data element is odd number
        for i in range(N):
            z = sort_vector[i] == mid_idx
            bit_vector.append(z)

        for i in range(N):
            median += shares[i] * bit_vector[i]
    else:           # When the number of data element is even number
        for i in range(N):
            mid_idx_m = mid_idx - 1
            z = (sort_vector[i] == mid_idx) + (sort_vector[i] == mid_idx_m)
            bit_vector.append(z)

        for i in range(N):
            median += shares[i] * bit_vector[i]
        
        median = median * (1/2)

    end_time = time.time()  # End timing
    time_cost = end_time - start_time # Time cost
    communication_cost = sy.comm_total / (2 ** 20)  # Communication cost in MB

    median_result = float(median.get().child)/pow(10, prec) # Rconstruct the result

    print("The median result: {}".format(median_result))
    print("The communication cost is {:.3f} MB".format(communication_cost))
    print("The time cost is {:.3f} ms".format(time_cost * 1000))


# Input
data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
secure_median(data)