import torch
import syft as sy
import sys
import math

sys.path.append('..')
import securefunc as sfc

hook = sy.TorchHook(torch)

def reciprocal_sqrt_newton_common(x):
    y = math.exp(-2.2*(x/2 + 0.2)) + 0.198046875
    print(y)
    for i in range(3):
        y = y * (3 - x * y * y) * 0.5
    print(y)

def reciprocal_sqrt_newton(x):
    alice = sy.VirtualWorker(hook, id="alice")
    bob = sy.VirtualWorker(hook, id="bob")
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")
    # y = int(0.198046875 * pow(10, prec))
    y = math.exp(-2.2*(x/2 + 0.2)) + 0.198046875
    y_sh = torch.tensor(y).fix_precision().share(bob,alice,crypto_provider=crypto_provider)
    x_sh = torch.tensor(x/2).fix_precision().share(bob,alice,crypto_provider=crypto_provider)
    
    for i in range(3):
        y_sh = y_sh * (1.5 - x_sh * y_sh * y_sh)
        # y_sh = sfc.__division(y_sh, 2, prec)
    z = y_sh * x_sh
    print(z.get()*2)

def sqrt_newton():
    alice = sy.VirtualWorker(hook, id="alice")
    bob = sy.VirtualWorker(hook, id="bob")

def secure_median():
    alice = sy.VirtualWorker(hook, id="alice")
    bob = sy.VirtualWorker(hook, id="bob")
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")
    numList = torch.Tensor([1, 2, 3, 4])
    shares = numList.fix_precision().share(alice, bob, crypto_provider=crypto_provider)
    # numList = [1, 2, 3, 4, 5]
    # shares = []
    # for num in numList:
    #     shares.append(torch.Tensor(num).fix_precision().share(alice, bob, crypto_provider=crypto_provider))

    print("======== Bits Vector Generation ========")
    sort_vector = []
    N = len(numList)

    for i in range(N):
        sum = shares[i] > shares[i]
        for j in range(N):
            if j == i:
                continue
            sum += shares[i] > shares[j]
        sort_vector.append(sum)

    bit_vector = []
    mid = torch.Tensor([N // 2]).fix_precision().share(alice, bob, crypto_provider=crypto_provider)
    print(mid.copy().get())
    
    if N % 2 == 1:
        for i in range(N):
            print(sort_vector[i].copy().get())
            z = sort_vector[i] == mid
            print(z.copy().get())
            bit_vector.append(z)
        
        print("======== Print Bits Vector ========")
        for i in range(N):
            print(bit_vector[i].copy().get())

        median = torch.tensor([0]).fix_precision().share(alice, bob, crypto_provider=crypto_provider)
        for i in range(N):
            median += shares[i] * bit_vector[i]
        
        print("======== Median ========")
        print(median.get())
    else:
        for i in range(N):
            mid_p = torch.Tensor([N // 2 - 1]).fix_precision().share(alice, bob, crypto_provider=crypto_provider)
            print(sort_vector[i].copy().get())
            z = (sort_vector[i] == mid) + (sort_vector[i] == mid_p)
            print(z.copy().get())
            bit_vector.append(z)
        
        print("======== Print Bits Vector ========")
        for i in range(N):
            print(bit_vector[i].copy().get())

        median = torch.tensor([0]).fix_precision().share(alice, bob, crypto_provider=crypto_provider)
        for i in range(N):
            median += shares[i] * bit_vector[i]
        
        median = median / 2

        print("======== Median ========")
        print(median.get())
if __name__ == "__main__":
    reciprocal_sqrt_newton(36)
    # secure_median()