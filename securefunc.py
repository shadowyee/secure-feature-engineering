import torch
import syft as sy
import sys

def isAdditiveShare(x):
    #TODO: try to ensure the variable has the type of shares
    # if isinstance(x, torch.Tensor) and torch.is_tensor(x.child) and isinstance(x.child, sy.AdditiveSharingTensor):
    if isinstance(x, torch.Tensor):
        return True
    return False

def secure_compute(x, y, method, prec):
    """
    Secure computation based on fss
    Only support two parties

    TODO: support more parties, the attrs have to be change like (tuple, method_name) to support more parties
    """

    if isAdditiveShare(x) and (isAdditiveShare(y) or isinstance(y, int)):
        name = "__secure_" + method
        func = getattr(sys.modules[__name__], name)
        if method == "div":
            return func(x, y, prec)
        else:
            return func(x, y)
    else:
        raise TypeError("Only additive share is allowed")

def __secure_add(x, y):
    """
    Add function based on secret sharing
    
    Here use the in-built ADD method in pysyft
    """
    return x + y

def __secure_eq(x, y):
    """
    Determine whether is two value are equal based on secret sharing
    
    Here use the in-built EQ method in pysyft
    """

    return x == y

def __secure_lt(x, y):
    """
    Determine whether is fisrt value is less than second value based on secret sharing

    Here use the in-built LT method in pysyft
    """

    return x < y

def __secure_mul(x, y):
    """
    Multipication function based on secret sharing

    Here use the in-built MUL method in pysyft
    """
    
    return x * y

def __secure_exp(x, y):
    """
    Exponentation function based on scret sharing

    Example:
        x, y has to be integer
        return x^y

    TODO: use fast exponentation to implement it
    """

    
    pass


def __secure_sqrt(x):
    """
    Square root operation based on secret sharing

    TODO: implement the method through the polynomial expansion
    """
    pass

def __secure_div(x, y, prec):
    """
    Division function based on secret sharing

    TODO: implement the method through the polynomial expansion

    Here we don't use the origin in-built div method in pysyft
    The in-built div method cost a lot
    """
    if isAdditiveShare(y):
        return x / y
    else:
        return __division(x, y, prec)

def __division(x, y, prec):
    """
    Implement division by multiplation and truncate
    """
    if not isAdditiveShare(x):
        raise AttributeError("The dividend is not Additive Share")
    if not isinstance(y, int):
        raise AttributeError("The divisor is not integer")
    
    multiplier = int((1 / y) * pow(10, prec))
    ret = x * multiplier

    return ret


def secure_reciprocal(x):
    """
    TODO: Calculate the reciprocal of x
    """
    pass

def __division_newton(a, b, precision=1e-3, max_iter=1000):
    """
    使用牛顿迭代法实现除法
    """
    # if b == 0:
    #     raise ValueError("除数不能为零")

    # 定义牛顿迭代函数
    def f(x):
        return a - (1 / b) * x

    # 定义牛顿迭代的导数函数
    def df(x):
        return -1 / b

    # 初始化迭代起点
    x = 1.0

    # 进行牛顿迭代
    for _ in range(max_iter):
        x_next = x - f(x) / df(x)

        x = x_next

    return x_next

#TODO: implement more methods althought they are not use in the current project

