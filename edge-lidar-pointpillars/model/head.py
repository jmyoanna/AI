# model/head.py
import torch
import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cls = nn.Conv2d(64, num_classes, 1)
        self.reg = nn.Conv2d(64, 7, 1)  # x,y,z,w,l,h,yaw

    def forward(self, x):
        return self.cls(x), self.reg(x)
