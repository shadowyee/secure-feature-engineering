import sys
sys.path.append('..')

import securefunc as sfc
import syft as sy
import torch

hook = sy.TorchHook(torch)

alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")

def get_worker_share(worker):
    total_size = 0
    objects = []
    for obj_id in worker._objects:
        obj = worker._objects[obj_id]
        if obj.shape == torch.Size([]):
            objects.append(int(obj))
        else:
            objects.append(tuple(obj.tolist()))
        total_size += obj.__sizeof__()
    return objects, total_size

def correction_test():
    """
    Test the correction of every function
    """
    # x = torch.tensor([25]).share(bob,alice)
    # y = torch.tensor([5]).share(bob,alice)

    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")   # 利用可信第三方生成乘法三元组
    x = torch.tensor([25]).share(bob,alice, crypto_provider=crypto_provider)
    y = torch.tensor([5]).share(bob,alice, crypto_provider=crypto_provider)

    # Add function
    add = sfc.secure_compute(x, y, "add")
    print("secure addition:", add.get())

    # Equal function
    eq = sfc.secure_compute(x, y, "eq")
    print("secure equation:", eq.get())

    # Less Than function
    lt = sfc.secure_compute(x, y, "lt")
    print("secure less than:", lt.get())

    # Multiplication function
    mul = sfc.secure_compute(x, y, "mul")
    print("secure mul:", mul.get())

    # TODO: Square Root function

    # Division function
    div = sfc.secure_compute(x, y, "div")
    print("secure div:", div.get())


def reciprocal_test():
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")   # 利用可信第三方生成乘法三元组
    x = torch.tensor([25]).share(bob,alice, crypto_provider=crypto_provider)
    z = sfc.secure_reciprocal(x)
    print(z.get())

def mean_test():
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")   # 利用可信第三方生成乘法三元组
    x = torch.tensor([1, 2, 3, 4]).share(bob,alice, crypto_provider=crypto_provider)
    s = x.sum()
    y = torch.tensor([4]).share(bob,alice, crypto_provider=crypto_provider)
    z = s * 1000 / y
    print(z.get())

def max_test():
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")   # 利用可信第三方生成乘法三元组
    x = torch.tensor(1).share(bob,alice, crypto_provider=crypto_provider)
    y = torch.tensor(2).share(bob,alice, crypto_provider=crypto_provider)
    z = torch.tensor(3).share(bob,alice, crypto_provider=crypto_provider)
    shares = [x, y, z]
    
    objects, objects_total_size = get_worker_share(alice)
    print(objects)
    
    ret = sfc.secure_max(shares)
    print(shares)
    shares = sfc.remove_share(shares, ret)
    print(shares)
    print(ret.get())

    # exec secure_max() thirdly
    ret = sfc.secure_max(shares)
    print(ret.get())

    objects, objects_total_size = get_worker_share(alice)
    print(objects)

if __name__ == "__main__":
    # print("========Correction Test========")
    # correction_test()

    # reciprocal_test()
    # mean_test()
    max_test()