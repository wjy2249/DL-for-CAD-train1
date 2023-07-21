import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from net import RNN
from torch.utils.data import DataLoader
from loss import bezier_curve,bezier_point
import scipy
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
print(torch.cuda.is_available())

d=3
num=100 #生成数据的规模，论文用1e6，我用100跑个小规模
dataset=[]
num_epochs = 30  # 迭代次数
batch_size = 4  # 批量大小

for cnt in range(0,num):
    c = np.random.standard_normal(size=(d + 1, 2))
    print(c) # d+1个控制点
    random_numbers = [random.random() for _ in range(2*d-1)]
    random_numbers.append(0)
    random_numbers.append(1)
    t=sorted(random_numbers)
    print(t) # 2d+1个曲线上的随机采样点的参数值
    P = []
    P2 = []
    edge = []

    for i in range(0,2*d+1):
        x,y=bezier_point(c,t[i])
        P.append([x,y])
    print(P) # 2d+1个参数值对应的坐标

    pxmin=P[0][0]
    pymin=P[0][0]
    pxmax=P[0][0]
    pymax=P[0][0]
    for i in range(1,2*d+1):
        pxmin=min(pxmin,P[i][0])
        pymin=min(pymin,P[i][1])
        pxmax=max(pxmax,P[i][0])
        pymax=max(pymax,P[i][1])
    Pmax=max(pxmax,pymax)
    Pmin=min(pxmin,pymin)


    for i in range(0,2*d+1):
        x=(P[i][0]-pxmin)/(Pmax-Pmin)
        y=(P[i][1]-pymin)/(Pmax-Pmin)
        P2.append([x,y])
    print(P2)

    for i in range(1,2*d+1):
        edge.append([P2[i][0]-P2[i-1][0],P2[i][1]-P2[i-1][1]])
    print(edge)
    dataset.append(edge)

print("\n\n")
print(dataset)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
model=RNN(2*d).to(device)
print(model)
print(2)

for data in dataset:
    print(data)
print("\n")
dataloader=DataLoader(dataset,batch_size=4,shuffle=False)
for inputs in dataloader:
    print(inputs,len(inputs))

print(1)
print(len(dataset),len(dataloader))
# 控制点坐标
control_points = np.array(dataset[0])
num_samples = 100  # 采样点数量

curve_points = bezier_curve(control_points, num_samples)

# 绘制曲线
plt.plot(control_points[:, 0], control_points[:, 1], 'ro-', label="Control Points")
plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label="Bezier Curve")
plt.legend()
plt.axis('equal')
plt.show()
