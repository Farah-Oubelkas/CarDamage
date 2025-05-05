import torch
import torch.nn as nn
import timm

class FeatureEncoder(nn.Module):
    def __init__(self, backbone='resnet50'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True)
    
    def forward(self, x):
        features = self.backbone(x)[-1]  # Use final feature map
        return features
