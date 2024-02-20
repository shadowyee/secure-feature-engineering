import torch
import syft as sy

hook = sy.TorchHook(torch)

class Parties():
    def __init__(self):
        self.data_owners = []
        self.crypto_provider = None

    def init_parties(self, owner_names, crypto_name):
        for people in owner_names:
            self.data_owners.append(sy.VirtualWorker(hook, id=people))

        self.crypto_provider = sy.VirtualWorker(hook, id=crypto_name) 

    def parties_info(self):
        print(self.data_owners)
        print(self.crypto_provider)
