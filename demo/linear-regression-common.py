import torch

def step_gradient(w_current:torch.Tensor, b_current:torch.Tensor, X:torch.Tensor, Y:torch.Tensor, lr):
    """
    计算误差函数在所有点上的导数，并更新w,b
    :param w_current: 当前的参数w
    :param b_current: 当前的参数b
    :param X:样本特征
    :param y:真实值
    :param lr:学习率
    :return:更新后的参数w,b
    """
    N = len(X) # 样本总个数
    
    # for i in range(N):
    #     x = X[i]
    #     y = Y[i]
    #     # 误差函数对w的导数，参考公式(3.5)
    #     mid = (2 / N) * ((w_current * x + b_current) - y) 
    #     b_gradient += mid
    #     # 误差函数对b的导数，参考公式(3.4)
    #     w_gradient += mid * x
    #     # w_gradient += b_gradient * x
    mid = (2 / N) * w_current * X + (2/N) * b_current - (2/N) * Y
    print(mid)
    b_gradient = mid.sum()
    w_gradient = torch.dot(mid, X)

    # 根据梯度下降算法更新参数w,b
    new_w = w_current - (lr * w_gradient)
    new_b = b_current - (lr * b_gradient)
    return [new_w, new_b]

def gradient_descent(X: torch.Tensor, Y: torch.Tensor, starting_w, starting_b, lr, num_iterations):
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
        w, b = step_gradient(w, b, X, Y, lr)
        # loss = mse(w, b, X, y)
        # if step % 100 == 0:
        #     print(f"iteration{step}, loss:{loss}, w:{w}, b:{b}")
        print(f"iteration{step}, w:{w}, b:{b}")
    
    return [w, b] # 返回最后一次的w, b

if __name__ == "__main__":
    X = torch.Tensor([1, 2, 3])
    Y = torch.Tensor([1, 2, 3])
    # X = [-1, 1]
    # Y = [0, 2.45]
    gradient_descent(X, Y, 0, 0, 0.01, 2)