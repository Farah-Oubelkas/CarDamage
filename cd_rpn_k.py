import torch
import torch.nn as nn
import torchvision.ops as ops

class CDRPNK(nn.Module):
    def __init__(self):
        super().__init__()
        # Use dummy RPN-like mechanism
        self.anchor_generator = ops.AnchorGenerator()

    def forward(self, feature_map):
        # Generate proposals (dummy implementation)
        proposals = torch.rand((feature_map.size(0), 100, 4))  # B, N, (x1, y1, x2, y2)
        return proposals
