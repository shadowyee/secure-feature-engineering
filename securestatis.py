"""
Use basic secure function to compute statistics
"""

import torch
import syft as sy
import numpy as np
import securefe as sfe
import securefunc as sfunc

def secure_mean(shares, dim, num):
    """
    """

    ret = []

    if dim == 1:
        for share in shares:
            sum = share.sum()
            mean = sum / num
        return {mean}

    elif dim == 2:
        mean = []
        for share in shares:
            s = share.sum()
            p = s
            r = p / num
            mean.append(r)

        return mean

    else:
        raise AttributeError("Secure Statistics Computation only support less than 3 dims")
        

def secure_varience(shares, mean, N):
    """
    """
    for m in mean:
        for i in range(N):
            mid = shares[i] - m
            if i == 0:
                ret = mid * mid
            else:
                ret += mid * mid
        return ret / N

def secure_std():
    """
    """

    pass

def secure_median(numList, hook, alice, bob, crypto_provider):
    """
    """
    shares = numList.fix_precision().share(alice, bob, crypto_provider=crypto_provider)
    print("Bits Vector Generation:")
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
    if N % 2 == 1:
        for i in range(N):
            z = sort_vector[i] == mid
            bit_vector.append(z)
        
        print("Print Bits Vector:")
        b_vec_print = []
        for i in range(N):
            b_vec_print.append(bit_vector[i].copy().get().child.child.item())
        print(b_vec_print)

        median = torch.tensor([0]).fix_precision().share(alice, bob, crypto_provider=crypto_provider)
        for i in range(N):
            median += shares[i] * bit_vector[i]
        
        return median
    else:
        for i in range(N):
            mid_p = torch.Tensor([N // 2 - 1]).fix_precision().share(alice, bob, crypto_provider=crypto_provider)
            z = (sort_vector[i] == mid) + (sort_vector[i] == mid_p)
            bit_vector.append(z)
        
        print("Print Bits Vector:")
        b_vec_print = []
        for i in range(N):
            b_vec_print.append(bit_vector[i].copy().get().child.child.item())
        print(b_vec_print)

        median = torch.tensor([0]).fix_precision().share(alice, bob, crypto_provider=crypto_provider)
        for i in range(N):
            median += shares[i] * bit_vector[i]
        
        median = median / 2
        return median
        
def secure_mode():
    """
    """
    pass

