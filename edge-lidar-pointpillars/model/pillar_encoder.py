# model/pillar_encoder.py
import torch
import torch.nn as nn

class PillarEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, x):
        # x: (P, N, 4)
        x = self.net(x)
        x = torch.max(x, dim=1)[0]  # PointNet 핵심
        return x
