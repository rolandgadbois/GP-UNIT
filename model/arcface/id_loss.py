import torch
from torch import nn

from model.simclr import SimCLR


class IDLoss(nn.Module):
    def __init__(self, checkpoint_path):
        super(IDLoss, self).__init__()
        print('Loading custom SimCLR encoder')
        self.encoder = SimCLR(checkpoint_path)
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.encoder.eval()

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop region like ArcFace
        x = self.face_pool(x)
        x_feats = self.encoder(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()

        loss = 0
        for i in range(n_samples):
            sim = torch.dot(y_hat_feats[i], y_feats[i])
            loss += 1 - sim
        return loss / n_samples
