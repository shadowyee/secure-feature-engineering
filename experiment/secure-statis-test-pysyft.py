import sys
sys.path.append('..')

import securestatis as sfs
import torch
import time
import syft as sy

def get_worker_share(worker):
    total_size = 0
    objects = []
    for obj_id in worker._objects:
        obj = worker._objects[obj_id]
        objects.append(tuple(obj.tolist()))
        total_size += obj.__sizeof__()
    return objects, total_size


print("===========Pysyft===========")
hook = sy.TorchHook(torch)

print("Create virtual participants:")
alice = sy.VirtualWorker(hook, id="alice")
print(alice)
bob = sy.VirtualWorker(hook, id="bob")
print(bob)
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")
print("---------------------------")

# # 求平均数
# # 输入为一维向量
print("Input:")
numList = torch.Tensor([1, 2, 3, 4])
print(numList)
shares = numList.fix_precision().share(alice, bob, crypto_provider=crypto_provider)
print("Shares owned by participants:")
objects, objects_total_size = get_worker_share(alice)
print("alice:", objects)
objects, objects_total_size = get_worker_share(bob)
print("bob:", objects)
mean_result = sfs.secure_mean({shares}, len(numList.shape), len(numList))
for res in mean_result:
    print("Mean:", int(res.get().child.child))

#输入为二维矩阵
print("Input:")
numList = torch.Tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
print(numList)
shares = []
for n in numList:
    shares.append(n.fix_precision().share(alice, bob, crypto_provider=crypto_provider))
print("Shares owned by participants:")
objects, objects_total_size = get_worker_share(alice)
print("alice:", objects)
objects, objects_total_size = get_worker_share(bob)
print("bob:", objects)
mean_result = sfs.secure_mean(shares, len(numList.shape), numList.shape[1])

result_list = []
for res in mean_result:
    result_list.append(int(res.get().child.child))
print("Mean:", result_list)
print("---------------------------")

print("Input:")
numList = torch.Tensor([1, 2, 3, 4, 5])
print(numList)
median_result = sfs.secure_median(numList, hook, alice, bob, crypto_provider)
objects, objects_total_size = get_worker_share(alice)
print("alice:", objects)
objects, objects_total_size = get_worker_share(bob)
print("bob:", objects)
print("Median:", median_result.get().child.child.item())
print("---------------------------")

# 方差
print("Input:")
numList = torch.Tensor([1, 2, 3, 4])
print(numList)
shares = numList.fix_precision().share(alice, bob, crypto_provider=crypto_provider)
dim = len(numList.shape)
N = len(numList)
objects, objects_total_size = get_worker_share(alice)
print("alice:", objects)
objects, objects_total_size = get_worker_share(bob)
print("bob:", objects)
varience_result = sfs.secure_varience(shares, sfs.secure_mean({shares}, dim, N), N)

print("Varience:",varience_result.get().child.child.item())
