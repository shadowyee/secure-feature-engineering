import torch
import syft as sy

def type_judge():
    #TODO: try to ensure the variable has the type of shares
    pass

def secure_add(x: torch.Tensor, y: torch.Tensor):
    """
    Add function based on secret sharing
    
    Here use the in-built ADD method in pysyft
    """

    return x + y

def secure_eq(x: torch.Tensor, y: torch.Tensor):
    """
    Determine whether is two value are equal based on secret sharing
    
    Here use the in-built EQ method in pysyft
    """

    return x == y

def secure_lt(x: torch.Tensor, y: torch.Tensor):
    """
    Determine whether is fisrt value is less than second value based on secret sharing

    Here use the in-built LT method in pysyft
    """

    return x < y

def secure_mul(x: torch.Tensor, y: torch.Tensor):
    """
    Multipication function based on secret sharing

    Here use the in-built MUL method in pysyft
    """
    
    return x * y

def secure_sqrt(x: torch.Tensor, y: torch.Tensor):
    """
    Square root operation based on secret sharing

    TODO: implement the method through the polynomial expansion
    """
    pass

def secure_div(x: torch.Tensor, y: torch.Tensor):
    """
    Division function based on secret sharing

    TODO: implement the method through the polynomial expansion

    Here we don't use the origin in-built div method in pysyft
    """
    pass

#TODO: implement more methods althought they are not use in the current project

