import numpy as np
import torch
from torch import nn


def bezier_curve(control_points, num_samples):
    t = np.linspace(0, 1, num_samples)
    n = len(control_points) - 1
    coefficients = np.array([binom(n, i) for i in range(n + 1)])
    curve_points = np.zeros((num_samples, 2))

    for i in range(num_samples):
        curve_points[i] = np.sum([control_points[j] * coefficients[j] * (t[i] ** j) * ((1 - t[i]) ** (n - j)) for j in range(n + 1)], axis=0)

    return curve_points


def binom(n, k):
    return np.math.factorial(n) // (np.math.factorial(k) * np.math.factorial(n - k))


def bernstein_poly(n, i, t):
    """
    计算Bernstein多项式的值
    """
    return np.math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))


def bezier_point(control_points, t):
    """
    计算贝塞尔曲线在参数值t处的坐标
    """
    n = len(control_points) - 1
    x = 0
    y = 0

    for i in range(n + 1):
        x += control_points[i][0] * bernstein_poly(n, i, t)
        y += control_points[i][1] * bernstein_poly(n, i, t)

    return x, y

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, predictions, targets,d,batch_size):
        sum=0
        for i in range(0,batch_size):
            for j in range(0,2*d+1):
                x=(predictions[i][j][0])-(targets[i][j][0])
                y=(predictions[i][j][1])-(targets[i][j][1])
                sum=sum+x*x+y*y
        return sum/(2*d+1)/batch_size
