# model.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights

num_joints = 29

class Pose3DNet(nn.Module):
    def __init__(self, num_joints):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, num_joints * 3),
        )

    def forward(self, x):
        out = self.backbone(x)
        return out.view(-1, num_joints, 3)
