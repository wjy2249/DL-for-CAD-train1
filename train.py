import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from net import RNN
from torch.utils.data import DataLoader
from loss import bezier_curve,bezier_point,Loss
import torch.optim as optim
import scipy
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
print(torch.cuda.is_available())

d=3 # 曲线的度d
num=100 #生成数据的规模，论文用1e6，我用1000跑个小规模
dataset=[]
trainpoint=[]
num_epochs = 30  # 迭代次数
batch_size = 4  # 批量大小
control_c=[]

for cnt in range(0,num):
    c = np.random.standard_normal(size=(d + 1, 2))
    control_c.append(c)
    print("c",c) # d+1个控制点
    random_numbers = [random.random() for _ in range(2*d-1)]
    random_numbers.append(0.0)
    random_numbers.append(1.0)
    t=sorted(random_numbers)
   # print(t) # 2d+1个曲线上的随机采样点的参数值
    P = []
    P2 = []
    edge = []

    for i in range(0,2*d+1):
        x,y=bezier_point(c,t[i])
        P.append([x,y])
  #  print(P) # 2d+1个参数值对应的坐标

    pxmin=min(P[0][0],P[0][1])
    pymin=min(P[0][0],P[0][1])
    pxmax=max(P[0][0],P[0][1])
    pymax=max(P[0][0],P[0][1])
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
        #print("P[i][0]",P[i][0],pxmin,Pmax-Pmin,P[i][1],pymin)
        P2.append([x,y])
   # print(P2)
    trainpoint.append(P2)

    for i in range(1,2*d+1):
        edge.append([P2[i][0]-P2[i-1][0],P2[i][1]-P2[i-1][1]])
  #  print(edge)
    dataset.append(edge)

print("\n\n")
print(trainpoint)


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

print(dataset)
criterion = Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

edges=torch.tensor(dataset,dtype=torch.float32).to(device)
labels=torch.tensor(trainpoint,dtype=torch.float32).to(device)

total_step = len(edges) // batch_size
print(edges.shape)
print("labels.shape",labels.shape)

matrix = torch.eye(2*d+1)
for i in range(1,2*d):
    matrix[i][i]=-1.0
#print(matrix)
repeated_matrix = matrix.unsqueeze(0).expand(num, -1, -1)
#print(repeated_matrix.shape)
b=torch.zeros(2*d+1)
b[2*d]=1

for epoch in range(num_epochs):
    all_p=[]
    for i in range(0, len(edges), batch_size):
        # 获取当前批次的数据
        batch_edges = edges[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        co_matrix=repeated_matrix[i:i+batch_size]

        # 前向传播
        outputs = model(batch_edges).to(device)
     #   print(co_matrix.shape,outputs.shape)
        print("outputs",outputs)
        for time in range(0, batch_size):
            for j in range(1,4*d-2,2):
                outputs[time][j]=1-outputs[time][j-1]

        print("outputs2",outputs)

        for time in range(0,batch_size):
            for j in range(1,2*d):
           # print(co_matrix[:batch_size,j,j-1],outputs[:batch_size,2*j-2])
                co_matrix[time,j,j-1]=outputs[time,2*j-2]
                co_matrix[time,j,j+1]=outputs[time,2*j-1]
            print("co_matrix",co_matrix[time])
            solve_t=torch.linalg.solve(co_matrix[time],b)
            print(solve_t)
            solve_p=[]
            for j in range(0,2*d+1):
                ppx,ppy=bezier_point(control_c[i+time],solve_t[j])
                pp=[ppx,ppy]
                solve_p.append(pp)
            tensor_p=torch.tensor(solve_p)
           # print("tensorp",tensor_p)
            all_p.append(tensor_p)

      #  print(co_matrix) #打印下系数矩阵
        prediction=all_p[0]
        prediction=prediction.unsqueeze(dim=0)
        prediction=prediction.repeat(batch_size,1,1).to(device)
        for time in range(1,batch_size):
            temp=all_p[time]
            prediction[time]=temp
      #  print("prediction",prediction,prediction.shape) #打印下参数化的点对应的坐标

        #反向传播和优化
      # print("train is beginning:",prediction,labels)
        loss=criterion(prediction,labels[i:i+batch_size],d,batch_size)
        loss.requires_grad_(True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印训练信息
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{i + 1}/{num_epochs}], Step [{i + 1}/{len(edges)}], Loss: {loss.item():.4f}')
