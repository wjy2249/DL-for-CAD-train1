import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Residualblock(nn.Module):
    def __init__(self,inn,hidden,out):
        super().__init__()
        self.Relu=nn.ReLU()
        self.residual_linear = nn.Linear(inn, out)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(inn, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out),
            nn.BatchNorm1d(out),
        )

    def forward(self, x):
        residual=self.residual_linear(inn)
        logits = self.linear_relu_stack(x)+residual
        logits=self.Relu()
        return logits

class RNN(nn.Module):
    def __init__(self,inn):
        super().__init__()
        self.block1=nn.Sequential(
            nn.Linear(inn,30),
            nn.ReLU(),
        )
        self.block2=nn.Sequential(
            Residualblock(30,50,70),
            Residualblock(70,70,70),
            Residualblock(70,70,70),
            Residualblock(70,50,30),
            Residualblock(30,30,30),
        )
        self.block3=nn.Sequential(
            nn.Linear(30,inn*2-2),
            nn.Sigmoid(),
        )

    def forward(self,x):
        logits=self.block1(x)
        logits=self.block2(logits)
        logits=self.block3(logits)
        return logits