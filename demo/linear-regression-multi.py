import torch
import syft as sy

hook = sy.TorchHook(torch)

alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

def step_gradient(w_current:torch.Tensor, b_current:torch.Tensor, X:torch.Tensor, Y:torch.Tensor, lr, N):
    """
    计算误差函数在所有点上的导数，并更新w,b
    :param w_current: 当前的参数w
    :param b_current: 当前的参数b
    :param X:样本特征
    :param y:真实值
    :param lr:学习率
    :return:更新后的参数w,b
    """
    
    # for i in range(N):
    #     x = X[i]
    #     y = Y[i]
    #     # 误差函数对w的导数，参考公式(3.5)
    #     mid = (2 / N) * ((w_current * x + b_current) - y) 
    #     b_gradient += mid
    #     # 误差函数对b的导数，参考公式(3.4)
    #     w_gradient += mid * x
    #     # w_gradient += b_gradient * x

    # print(w_current.copy().get())
    # print(b_current.copy().get())
    mid = (w_current * X + b_current - Y) * 2
    mid /= N
    # print(mid.copy().get())
    
    b_gradient = mid.sum()
    w_gradient = torch.dot(mid, X)
    # print(b_gradient.copy().get())
    # print(w_gradient.copy().get())

    # 根据梯度下降算法更新参数w,b
    # b_mid = (lr * b_gradient)
    b_mid = b_gradient / 100
    # print("b_mid", b_mid.copy().get())
    new_b = b_current - b_mid

    # w_mid = (lr * w_gradient)
    w_mid = w_gradient / 100
    # print("w_mid", w_mid.copy().get())
    new_w = w_current - w_mid
    return [new_w, new_b]

def gradient_descent(X: torch.Tensor, Y: torch.Tensor, starting_w, starting_b, lr, num_iterations, N):
    """
    更新num_iterations个Epoch后的w,b
    :param X: 样本特征
    :param y: 真实值
    :param starting_w: 初始的参数w 
    :param starting_b: 初始的参数b
    :param lr: 学习率
    :param num_iterations: 迭代的次数
    :return: 返回最后一次梯度下降算法后的参数w, b
    """
    w = starting_w
    b = starting_b
    # 迭代更新num_iterations次参数更新
    for step in range(num_iterations):
        # 计算一次梯度下降算法更新参数
        w, b = step_gradient(w, b, X, Y, lr, N)
        # loss = mse(w, b, X, y)
        # if step % 100 == 0:
        #     print(f"iteration{step}, loss:{loss}, w:{w}, b:{b}")

        print("===========", step, "===========")
        # print(f"iteration{step}, w:{w}, b:{b}")
        print(w.copy().get(), b.copy().get())

if __name__ == "__main__":
    N = 3
    prec = 7
    X = torch.Tensor([1, 2, 3]).fix_precision(precision_fractional=prec).share(alice, bob, crypto_provider=crypto_provider, protocol="fss")
    Y = torch.Tensor([1, 2, 3]).fix_precision(precision_fractional=prec).share(alice, bob, crypto_provider=crypto_provider, protocol="fss")
    # X = [-1, 1]
    # Y = [0, 2.45]
    w_start = torch.Tensor([0]).fix_precision(precision_fractional=prec).share(alice, bob, crypto_provider=crypto_provider, protocol="fss")
    b_start = torch.Tensor([0]).fix_precision(precision_fractional=prec).share(alice, bob, crypto_provider=crypto_provider, protocol="fss")
    
    gradient_descent(X, Y, w_start, b_start, 0.01, 2000, N)