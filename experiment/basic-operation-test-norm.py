import crypten
import torch
import time
import csv
import math
import random

crypten.init()

print("===========Norm===========")
x = random.randint(1, 100)
y = random.randint(1, 100)

start_time = time.time()
result_add = x + y
end_time = time.time()
add_time = end_time - start_time
print("ADD time:", add_time)

start_time = time.time()
result_mul = x * y
end_time = time.time()
mul_time = end_time - start_time
print("MUL time:", mul_time)

start_time = time.time()
result_div = x / y
end_time = time.time()
div_time = end_time - start_time
print("DIV time:", div_time)

start_time = time.time()
result_sqrt = math.sqrt(x)
end_time = time.time()
sqrt_time = end_time - start_time
print("SQRT time:", sqrt_time)

start_time = time.time()
result_compare = x > y
end_time = time.time()
compare_time = end_time - start_time
print("CMP time:", compare_time)

with open('operation_times.csv', mode='r+', newline='') as file:
    writer = csv.writer(file)
    reader = csv.reader(file)
    header = next(reader, None)
    if not header:
        writer.writerow(["Library", "ADD", "MUL", "DIV", "SQRT", "CMP"])

add_time = "{:.7f}".format(add_time)
mul_time = "{:.7f}".format(mul_time)
div_time = "{:.7f}".format(div_time)
sqrt_time = "{:.7f}".format(sqrt_time)
compare_time = "{:.7f}".format(compare_time)

with open('operation_times.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Norm", add_time, mul_time, div_time, sqrt_time, compare_time])