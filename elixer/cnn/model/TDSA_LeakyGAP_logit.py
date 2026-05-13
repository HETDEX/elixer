import torch
import torch.nn as nn
import torch.nn.functional as F

# TDSA: Two-Dimensional Spectrum Architecture
# LeakyGAP: using LeakyReLU and Global Average Pooling
# logit: Return logits, not probabilities
class TDSA_LeakyGAP_logit(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, dropout_rate: float):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout2d(dropout_rate)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout2d(dropout_rate),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        self.classifier = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(32, num_classes)  
        )

    def forward(self, x):
        x = self.block1(x)         # (B, 16, 9, 40)
        x = self.block2(x)         # (B, 16, 9, 40)
        
        x = self.gap(x)            # (B, 16, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 16)
        x = self.classifier(x)
        #x = torch.sigmoid(x) # Do not apply sigmoid here; return logits instead of probabilities
        return x

