import sys
sys.path.append('..')

import securestatis as sst
import participants as pt
import syft as sy
import torch

def secure_mean_test():
    data = torch.tensor([1, 2, 3, 4, 5, 6])
    data_owners = []
    parties_num = 3
    for i in range(0, parties_num):
        data_owners.append("workers" + str(i))
    crypto_provider = "crypro_provider"

    parties = pt.Parties()
    parties.init_parties(data_owners, crypto_provider)

    fix_prec = 3
    prec = 5
    mean = sst.secure_mean(parties.data_owners, parties.crypto_provider, data, prec)

    print("The mean of 1 dim data:", float(mean.get())/pow(10, fix_prec + prec))

if __name__ == "__main__":
    secure_mean_test()