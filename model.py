import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
import random

# --- Seed setup function ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 1. Simple YAMNet-like model ---
class SimpleYAMNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleYAMNet, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Input: [B, 1, 96, 64]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [B, 16, 48, 32]

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [B, 32, 24, 16]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # -> [B, 64, 1, 1]
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  
        out = self.classifier(x)
        return out

# --- 2. RawNet-like model ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class RawNet(nn.Module):
    def __init__(self, num_classes=2):
        super(RawNet, self).__init__()
        self.frontend = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=3, padding=1),  # → [B, 64, ~21333]
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, downsample=True),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, downsample=True),
            ResidualBlock(256, 256)
        )
        self.gru = nn.GRU(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):  # x: [B, 1, 64000]
        x = self.frontend(x)           # → [B, 64, T]
        x = self.res_blocks(x)         # → [B, 256, T]
        x = x.permute(0, 2, 1)         # → [B, T, 256]
        _, h_n = self.gru(x)           # h_n: [2, B, 128]
        h = torch.cat([h_n[0], h_n[1]], dim=1)  # → [B, 256]
        return self.classifier(h)

# --- 3. LFCC_LCNN model with Max-Feature-Map ---
class MFM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(MFM, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv(x)
        out = torch.split(x, self.out_channels, dim=1)
        return torch.max(out[0], out[1])

class LFCC_LCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(LFCC_LCNN, self).__init__()
        self.layer1 = MFM(1, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer2 = MFM(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer3 = MFM(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer4 = MFM(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):  # [B, 1, T, D] e.g. [B, 1, 100, 60]
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        x = self.pool4(self.layer4(x))  # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)       # Flatten → [B, 128]
        return self.classifier(x)

# --- 4. LFCC_GMR model using Gaussian Mixture Models ---
class LFCC_GMR:
    def __init__(self, n_components=2):
        self.gmm0 = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=42)
        self.gmm1 = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=42)

    def fit(self, X0, X1):
        self.gmm0.fit(X0)
        self.gmm1.fit(X1)

    def predict(self, X):
        scores0 = self.gmm0.score_samples(X)
        scores1 = self.gmm1.score_samples(X)
        pred = 0 if np.mean(scores0) > np.mean(scores1) else 1
        return pred