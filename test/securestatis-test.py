import sys
sys.path.append('..')

import securestatis as sst
import participants as pt
import syft as sy
import torch

hook = sy.TorchHook(torch)

alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

def secure_mean_test():
    # data = torch.tensor([1, 2, 3, 4, 5, 6])
    # data = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    data = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7]])
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

    for m in mean:
        print("The mean of data:", float(m.get())/pow(10, fix_prec + prec))        

def secure_median_test():
    a = torch.tensor(1).share(bob,alice, crypto_provider=crypto_provider)
    b = torch.tensor(2).share(bob,alice, crypto_provider=crypto_provider)
    c = torch.tensor(3).share(bob,alice, crypto_provider=crypto_provider)
    d = torch.tensor(4).share(bob,alice, crypto_provider=crypto_provider)
    
    shares = [a, b, c, d]
    print(sst.secure_median(shares))

if __name__ == "__main__":
    # secure_mean_test()
    secure_median_test()