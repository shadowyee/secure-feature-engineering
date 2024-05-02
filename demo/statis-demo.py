import torch
import syft as sy
import sys
import math

sys.path.append('..')
import securefunc as sfc

hook = sy.TorchHook(torch)

def reciprocal_sqrt_newton_common(x):
    y = math.exp(-2.2*(x/2 + 0.2)) + 0.198046875
    print(y)
    for i in range(3):
        y = y * (3 - x * y * y) * 0.5
    print(y)

def reciprocal_sqrt_newton(x):
    alice = sy.VirtualWorker(hook, id="alice")
    bob = sy.VirtualWorker(hook, id="bob")
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")
    # y = int(0.198046875 * pow(10, prec))
    y = math.exp(-2.2*(x/2 + 0.2)) + 0.198046875
    y_sh = torch.tensor(y).fix_precision().share(bob,alice,crypto_provider=crypto_provider)
    x_sh = torch.tensor(x/2).fix_precision().share(bob,alice,crypto_provider=crypto_provider)
    for i in range(3):
        y_sh = y_sh * (1.5 - x_sh * y_sh * y_sh)
        # y_sh = sfc.__division(y_sh, 2, prec)
    print(y_sh.get())

def sqrt_newton():
    alice = sy.VirtualWorker(hook, id="alice")
    bob = sy.VirtualWorker(hook, id="bob")

if __name__ == "__main__":
    reciprocal_sqrt_newton(36)
