import sys
sys.path.append('..')

import securestatis as sst
import securefe as sfe
import securefunc as sfunc
import numpy as np
import participants as pt
import syft as sy
import torch

data_owners = None
crypto_provider = None
data = None

def file_to_tensor(file_path):
    """

    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    matrix = []
    for line in lines:
        row = [float(x) for x in line.split()]  
        matrix.append(row)

    tensor = torch.tensor(matrix, dtype=torch.float32)

    return tensor

def divide_into_shares():
    """

    """
    file_path = 'uploads/test.txt'  

    global data
    data = file_to_tensor(file_path)

    parties_num = 8
    owner_names = []
    for i in range(0, parties_num):
        owner_names.append("workers" + str(i))

    parties = pt.Parties()
    parties.init_parties(owner_names, "crypro_provider")

    global data_owners, crypro_provider
    data_owners = parties.get_parties()
    crypro_provider = parties.get_cryptopvd()

    dim = len(data.shape)
    shares = []
    if dim == 1:
        shares = {sfe.secret_share(data, data_owners, crypto_provider, False)}

    elif dim == 2:
        data = np.transpose(data)
        for d in data:
            shares.append(sfe.secret_share(d, data_owners, crypto_provider, False))

    return shares

def secure_mean_compute(shares):
    """

    """
    # data = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7]])

    fix_prec = 3
    prec = 5
    # mean = sst.secure_mean(parties.data_owners, parties.crypto_provider, data, prec)

    global data
    dim = len(data.shape)
    
    ret = []

    if dim == 1:
        num = data.shape[0]
        for share in shares:
            sum = share.sum()
            mean = sfunc.secure_compute(sum, num, "div", prec)
            ret = {float(mean.get())/pow(10, fix_prec + prec)}

    elif dim == 2:
        data = np.transpose(data)
        mean = []
        for share in shares:
            mean.append(sfunc.secure_compute(share.sum(), share.shape[0], "div", prec))

        for m in mean:
            # print("The mean of data:", float(m.get())/pow(10, fix_prec + prec))
            ret.append(float(m.get())/pow(10, fix_prec + prec))

    print(ret)

def get_middle_result():
    """
    Get the shares every worker has.
    """
    def get_worker_share(worker):
        total_size = 0
        objects = []
        for obj_id in worker._objects:
            obj = worker._objects[obj_id]
            objects.append(obj)
            total_size += obj.__sizeof__()
        return objects, total_size

    global data_owners
    for idx, i in enumerate(data_owners):
        objects, objects_total_size = get_worker_share(i)
        print(f"Local Worker {idx}: {objects} objects, {objects_total_size} bytes")

if __name__ == "__main__":
    shares = None
    attr_num = len(sys.argv)
    if attr_num == 1:
        raise AttributeError("No argument.")
    elif attr_num > 2:
        raise AttributeError("Too much arguments.")
    else:
        if sys.argv[1] == "-d":
            shares = divide_into_shares()
        elif sys.argv[1] == "-m":
            get_middle_result()
        elif sys.argv[1] == "-c":
            if shares == None:
                raise Exception("No shares in local workers.")
            else:
                secure_mean_compute(shares)
        else:
            raise AttributeError("Arg Error!")
 
