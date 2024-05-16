import torch
import syft as sy
import sys

hook = sy.TorchHook(torch)

alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")

def get_worker_share(worker):
    total_size = 0
    objects = []
    for obj_id in worker._objects:
        obj = worker._objects[obj_id]
        objects.append(tuple(obj.tolist()))
        total_size += obj.__sizeof__()
    return objects, total_size

loop_num = 1000
for i in range(loop_num):
    y = torch.tensor([5]).share(bob,alice)
    objects, objects_total_size = get_worker_share(alice)
    print(objects[0][0])