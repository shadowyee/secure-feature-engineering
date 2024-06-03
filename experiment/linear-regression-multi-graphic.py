import matplotlib.pyplot as plt
import csv

# 假设有两组坐标
time_pysyft = []
cost_pysyft = []

time_common = []
cost_common = []

with open('linear-multi-pysyft.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        time_pysyft.append(float(row[0]))
        cost_pysyft.append(float(row[1]))


with open('linear-multi-common.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        time_common.append(float(row[0]))
        cost_common.append(float(row[1]))

# 创建一个图形和一个子图
plt.figure()
plt.subplot()

# 在子图中绘制两条曲线，使用不同的线型或标记
plt.plot(time_pysyft, cost_pysyft, label='Ours', linestyle='-')
plt.plot(time_common, cost_common, label='Norm', linestyle='--')

# 添加图例、标签等信息
plt.xlabel('Time(seconds)')
plt.ylabel('Cost')
plt.title('Multivariable Linear Regression Performance Comparison')
plt.legend()

plt.savefig('linear_regression_multi_bw.png')

# 显示图形
plt.show()
