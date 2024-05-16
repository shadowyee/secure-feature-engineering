import csv
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件并提取数据
data = {}
with open('operation_times.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        library = row['Library']
        data[library] = {k: float(v) for k, v in row.items() if k != 'Library'}

# 绘制柱状图
labels = list(data['Crypten'].keys())
crypten_values = list(data['Crypten'].values())
norm_values = list(data['Norm'].values())
pysyft_values = list(data['Pysyft'].values())

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots()
rects1 = ax.bar(x, norm_values, width, label='Math', color='gray', edgecolor='black')
rects2 = ax.bar(x + width, crypten_values, width, label='Crypten', color='gray', edgecolor='black', hatch='//')
rects3 = ax.bar(x + 2*width, pysyft_values, width, label='Ours', color='gray', edgecolor='black', hatch='\\')

ax.set_ylabel('Time (seconds)')
ax.set_title('Basic Operation Time')
ax.set_xticks(x + width)
ax.set_xticklabels(labels)
ax.legend()

# 设置对数坐标轴
ax.set_yscale('log')

plt.savefig("operation_times_chart_bw.png")

plt.show()
