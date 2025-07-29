import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SimCLR(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.encoder = models.resnet18(pretrained=False)
        self.encoder.fc = nn.Identity()
        self.encoder.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.encoder.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            x = F.normalize(x, dim=1)
        return x