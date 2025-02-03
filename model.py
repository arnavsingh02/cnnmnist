import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(28**2, 256)
        self.Matrix2 = nn.Linear(256, 128)
        self.Matrix3 = nn.Linear(128, 64)
        self.Matrix4 = nn.Linear(64, 10)
        self.R = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(-1, 28**2)
        x = self.R(self.Matrix1(x))
        x = self.dropout(x)
        x = self.R(self.Matrix2(x))
        x = self.dropout(x)
        x = self.R(self.Matrix3(x))
        x = self.dropout(x)
        x = self.Matrix4(x)
        return x.squeeze()
