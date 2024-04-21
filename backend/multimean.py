import sys
sys.path.append('..')

import securestatis as sst
import participants as pt
import syft as sy
import torch

def file_to_tensor(file_path):

    with open(file_path, 'r') as file:
        lines = file.readlines()

    matrix = []
    for line in lines:
        row = [float(x) for x in line.split()]  
        matrix.append(row)

    tensor = torch.tensor(matrix, dtype=torch.float32)

    return tensor

def secure_mean_compute():
    # data = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7]])
    file_path = 'uploads/test.txt'  
    data = file_to_tensor(file_path)

    data_owners = []
    parties_num = 8
    for i in range(0, parties_num):
        data_owners.append("workers" + str(i))
    crypto_provider = "crypro_provider"

    parties = pt.Parties()
    parties.init_parties(data_owners, crypto_provider)

    fix_prec = 3
    prec = 5
    mean = sst.secure_mean(parties.data_owners, parties.crypto_provider, data, prec)

    ret = []
    for m in mean:
        # print("The mean of data:", float(m.get())/pow(10, fix_prec + prec))
        ret.append( float(m.get())/pow(10, fix_prec + prec))

    print(ret)

if __name__ == "__main__":
    secure_mean_compute()

