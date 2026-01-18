# main.py
import numpy as np
import torch
from utils.pillar import create_pillars
from model.pillar_encoder import PillarEncoder
from model.backbone import BEVBackbone
from model.head import DetectionHead
from config import *

# Load fake LiDAR
points = np.load("data/fake_lidar.npy")

pillars, coords = create_pillars(points)
pillars = torch.tensor(pillars, dtype=torch.float32)

# FP16 실험도 가능
use_fp16 = False
if use_fp16:
    pillars = pillars.half()

encoder = PillarEncoder()
backbone = BEVBackbone()
head = DetectionHead(NUM_CLASSES)

if use_fp16:
    encoder = encoder.half()
    backbone = backbone.half()
    head = head.half()

pillar_features = encoder(pillars)

# BEV Map 만들기 (단순 버전)
bev = torch.zeros((1, 64, 40, 40))
for feat, (x, y) in zip(pillar_features, coords):
    bev[0, :, x % 40, y % 40] = feat

bev_feat = backbone(bev)
cls, reg = head(bev_feat)

print("Cls shape:", cls.shape)
print("Reg shape:", reg.shape)
