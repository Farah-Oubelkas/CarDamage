import torch
import torch.nn as nn

class SwAVModule(nn.Module):
    def __init__(self, feature_dim=256, num_prototypes=3000):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))

    def forward(self, z):
        # Normalize features
        z = nn.functional.normalize(self.projection(z), dim=1)
        assignments = torch.matmul(z, self.prototypes.T)
        return assignments
