"""
Use basic secure function to compute statistics
"""

import torch
import syft as sy
import numpy as np
import securefe as sfe
import securefunc as sfunc

def secure_mean(workers, crypto_provider, data, prec):
    """
    """
    if not isinstance(data, torch.Tensor):
        raise AttributeError("Attr data has to be Tensor")
    dim = len(data.shape)

    if dim == 1:
        # TODO: fixed precision
        num = data.shape[0]
        shares = sfe.secret_share(data, workers, crypto_provider, False)
        
        # use torch.mean() will cause unexpected error
        # mean = torch.mean(shares, dim=0)
        # divide num will cause unexpected error
        # mean = shares.sum() / num
        # if the num of parties is more than 2 will cause the above problems
        
        # TODO: the division method in securenn is not worked
        # mean = securenn.division(shares, num)

        sum = shares.sum()
        mean = sfunc.secure_compute(sum, num, "div", prec)
        return {mean}

    elif dim == 2:
        data = np.transpose(data)
        ret = []
        for d in data:
            shares = sfe.secret_share(d, workers, crypto_provider, False)
            ret.append(sfunc.secure_compute(shares.sum(), shares.shape[0], "div", prec))
        return ret
        
    else:
        raise AttributeError("Secure Statistics Computation only support less than 3 dims")
        

def secure_varience():
    """
    """
    pass

def secure_std():
    """
    """
    pass

def secure_median():
    """
    """
    pass

def secure_mode():
    """
    """
    pass

