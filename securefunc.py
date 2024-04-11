import torch
import syft as sy
import sys

def isAdditiveShare(x):
    #TODO: try to ensure the variable has the type of shares
    if isinstance(x, torch.Tensor) and torch.is_tensor(x.child) and isinstance(x.child, sy.AdditiveSharingTensor):
        return True
    return False

def secure_compute(x, y, method):
    """
    Secure computation based on fss
    Only support two parties

    TODO: support more parties, the attrs have to be change like (tuple, method_name) to support more parties
    """
    if isAdditiveShare(x) and isAdditiveShare(y):
        func = getattr(sys.modules[__name__], method)
        return func(x, y)
    else:
        raise TypeError("Only addtive share is allowed")

def secure_add(x, y):
    """
    Add function based on secret sharing
    
    Here use the in-built ADD method in pysyft
    """
    return x + y

def secure_eq(x, y):
    """
    Determine whether is two value are equal based on secret sharing
    
    Here use the in-built EQ method in pysyft
    """

    return x == y

def secure_lt(x, y):
    """
    Determine whether is fisrt value is less than second value based on secret sharing

    Here use the in-built LT method in pysyft
    """

    return x < y

def secure_mul(x, y):
    """
    Multipication function based on secret sharing

    Here use the in-built MUL method in pysyft
    """
    
    return x * y

def secure_sqrt(x):
    """
    Square root operation based on secret sharing

    TODO: implement the method through the polynomial expansion
    """
    pass

def secure_div(x, y):
    """
    Division function based on secret sharing

    TODO: implement the method through the polynomial expansion

    Here we don't use the origin in-built div method in pysyft
    """
    return x / y

#TODO: implement more methods althought they are not use in the current project

