import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import syft as sy

# 初始化 PySyft
hook = sy.TorchHook(torch)
alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")

# 生成数据
data = []

def get_worker_share(worker):
    total_size = 0
    objects = []
    for obj_id in worker._objects:
        obj = worker._objects[obj_id]
        objects.append(tuple(obj.tolist()))
        total_size += obj.__sizeof__()
    return objects, total_size

loop_num = 1000000
for i in range(loop_num):
    y = torch.tensor([5]).share(bob, alice)
    objects, objects_total_size = get_worker_share(alice)
    data.append(objects[0][0])

# 将数据转换为 NumPy 数组
data = np.array(data)

# 使用 Matplotlib 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(data, bins=100, edgecolor='black', alpha=0.7)
plt.title('Distribution of Secret Shares')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig('distribution-of-secret-shares.png')
plt.show()

# # 使用 Matplotlib 绘制散点图
# plt.figure(figsize=(10, 6))
# plt.scatter(np.arange(len(data)), data, alpha=0.5)
# plt.title('Scatter Plot of Data')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.savefig('num2.png')
# plt.show()
