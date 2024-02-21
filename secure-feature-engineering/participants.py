import torch
import syft as sy

class Parties():
    def __init__(self):
        self.data_owners = []
        self.crypto_provider = None
        self.hook = sy.TorchHook(torch)

    def init_parties(self, owner_names, crypto_name):
        """
        Initialize the data onwers and crypto provider
        """
        for people in owner_names:
            self.data_owners.append(sy.VirtualWorker(self.hook, id=people))

        self.crypto_provider = sy.VirtualWorker(self.hook, id=crypto_name) 

    def parties_info(self):
        """
        Print all parties information
        """
        print(self.data_owners)
        print(self.crypto_provider)
