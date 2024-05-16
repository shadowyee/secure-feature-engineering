import sys
sys.path.append('..')

import securefunc as sfc
import torch
import time
import csv
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
# 创建虚拟参与方
print("Create virtual participants:")
hook = sy.TorchHook(torch)
alice = sy.VirtualWorker(hook, id="alice")
print(alice)
bob = sy.VirtualWorker(hook, id="bob")
print(bob)
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")
print("---------------------------")

# 输入
print("Input:")
x = torch.tensor([2])
y = torch.tensor([3])
print(x)
print(y)
print("---------------------------")

# 秘密共享
print("Secret share pointers:")
x_encrypted = x.fix_precision().share(alice, bob, crypto_provider=crypto_provider)
y_encrypted = y.fix_precision().share(alice, bob, crypto_provider=crypto_provider)
print(x_encrypted)
print(y_encrypted)
print("Shares owned by participants:")
objects, objects_total_size = get_worker_share(alice)
print("alice:", objects)
objects, objects_total_size = get_worker_share(bob)
print("bob:", objects)
print("---------------------------")

# 加法
start_time = time.time()
result_add = sfc.__secure_add(x_encrypted, y_encrypted)
end_time = time.time()
add_time = end_time - start_time
print("ADD time:{:.5f}s".format(add_time))
objects, objects_total_size = get_worker_share(alice)
print("alice:", objects)
objects, objects_total_size = get_worker_share(bob)
print("bob:", objects)
print("ADD result:", result_add.get().child.child)
print("---------------------------")

# 乘法
start_time = time.time()
result_mul = sfc.__secure_mul(x_encrypted, y_encrypted)
end_time = time.time()
mul_time = end_time - start_time
print("MUL time:{:.5f}s".format(mul_time))
objects, objects_total_size = get_worker_share(alice)
print("alice:", objects)
objects, objects_total_size = get_worker_share(bob)
print("bob:", objects)
print("MUL result:", result_mul.get().chlid.child)
print("---------------------------")

start_time = time.time()
result_compare = sfc.__secure_eq(x_encrypted, y_encrypted)
end_time = time.time()
compare_time = end_time - start_time
print("CMP time:{:.5f}s".format(compare_time))
objects, objects_total_size = get_worker_share(alice)
print("alice:", objects)
objects, objects_total_size = get_worker_share(bob)
print("bob:", objects)
print("CMP result:", result_compare.get().child.child)
print("---------------------------")


# 针对常量运算的优化后除法
start_time = time.time()
result_sdiv = sfc.__division(x_encrypted, 2, 3)
end_time = time.time()
sdiv_time = end_time - start_time
print("s-DIV time:{:.5f}s".format(sdiv_time))
objects, objects_total_size = get_worker_share(alice)
print("alice:", objects)
objects, objects_total_size = get_worker_share(bob)
print("bob:", objects)
print("s-DIV result:", result_sdiv.get().child.child/pow(10,3))
print("---------------------------")

# 除法
start_time = time.time()
result_div = sfc.__secure_div(x_encrypted, y_encrypted)
end_time = time.time()
div_time = end_time - start_time
print("DIV time:{:.5f}s".format(div_time))
objects, objects_total_size = get_worker_share(alice)
print("alice:", objects)
objects, objects_total_size = get_worker_share(bob)
print("bob:", objects)
print("DIV result:", result_div.get().child.child)
print("---------------------------")


# 开方运算
server1 = sy.VirtualWorker(hook, id="server1")
server2 = sy.VirtualWorker(hook, id="server2")

start_time = time.time()
result_sqrt = sfc.__secure_sqrt(int(x), server1, server2, crypto_provider)
end_time = time.time()
sqrt_time = end_time - start_time
print("SQRT time:{:.5f}s".format(sqrt_time))
objects, objects_total_size = get_worker_share(server1)
print("alice:", objects)
objects, objects_total_size = get_worker_share(server2)
print("bob:", objects)
print("SQRT result:", result_sqrt.get().child.child)
print("---------------------------")

exit()
with open('operation_times.csv', mode='r+', newline='') as file:
    writer = csv.writer(file)
    reader = csv.reader(file)
    header = next(reader, None)
    if not header:
        writer.writerow(["Library", "ADD", "MUL", "DIV", "SQRT", "CMP"])

add_time = round(add_time, 7)
mul_time = round(mul_time, 7)
div_time = round(div_time, 7)
sqrt_time = round(sqrt_time, 7)
compare_time = round(compare_time, 7)

with open('operation_times.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Pysyft", add_time, mul_time, div_time, sqrt_time, compare_time])