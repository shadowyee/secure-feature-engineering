import crypten
import torch
import time
import csv

crypten.init()

print("===========Crypten===========")
x_encrypted = crypten.cryptensor(torch.tensor([1.0]))
y_encrypted = crypten.cryptensor(torch.tensor([4.0]))

start_time = time.time()
result_add = x_encrypted + y_encrypted
end_time = time.time()
add_time = end_time - start_time
print("ADD time:", add_time)

start_time = time.time()
result_mul = x_encrypted * y_encrypted
end_time = time.time()
mul_time = end_time - start_time
print("MUL time:", mul_time)

start_time = time.time()
result_div = x_encrypted / y_encrypted
end_time = time.time()
div_time = end_time - start_time
print("DIV time:", div_time)

start_time = time.time()
result_sqrt = x_encrypted.sqrt()
end_time = time.time()
sqrt_time = end_time - start_time
print("SQRT time:", sqrt_time)

start_time = time.time()
result_compare = x_encrypted > y_encrypted
end_time = time.time()
compare_time = end_time - start_time
print("CMP time:", compare_time)

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
    writer.writerow(["Crypten", add_time, mul_time, div_time, sqrt_time, compare_time])