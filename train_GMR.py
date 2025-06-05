import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import os
import torchaudio
from torchaudio.transforms import LFCC
from glob import glob

# --- Model Definition: LFCC_LCNN with Max-Feature-Map ---
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

    def forward(self, x):  # [B, 1, T, D]
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        x = self.pool4(self.layer4(x))  # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)       # Flatten → [B, 128]
        return self.classifier(x)

# --- Custom Dataset with LFCC feature extraction and Padding ---
class AudioAttackDataset(Dataset):
    def __init__(self, root_dir, max_len=200, sample_rate=16000, n_lfcc=60):
        self.data = []
        self.labels = []
        self.max_len = max_len
        self.sample_rate = sample_rate
        self.transform = LFCC(sample_rate=sample_rate, n_lfcc=n_lfcc)

        label_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.label_map = {label: idx for idx, label in enumerate(label_names)}
        self.num_classes = len(label_names)

        print(f"Found classes: {self.label_map}")

        for label in label_names:
            files = glob(os.path.join(root_dir, label, "*.flac"))
            for file_path in files:
                waveform, sr = torchaudio.load(file_path)
                if sr != sample_rate:
                    resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                    waveform = resample(waveform)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                lfcc = self.transform(waveform)
                lfcc = lfcc.permute(0, 2, 1)  # [1, time, freq]

                self.data.append(lfcc)
                self.labels.append(self.label_map[label])

    def __getitem__(self, idx):
        x, y = self.data[idx], self.labels[idx]
        T = x.shape[1]
        if T < self.max_len:
            x = F.pad(x, (0, 0, 0, self.max_len - T))
        elif T > self.max_len:
            x = x[:, :self.max_len, :]
        return x, y

    def __len__(self):
        return len(self.data)

# --- Training function ---
def train_lcnn_classifier(train_loader, val_loader, device, epochs=10):
    num_classes = train_loader.dataset.dataset.num_classes if isinstance(train_loader.dataset, torch.utils.data.Subset) else train_loader.dataset.num_classes
    model = LFCC_LCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1} — Loss: {total_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                out_val = model(x_val)
                preds_val = out_val.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds_val)
                all_labels.extend(y_val.numpy())

        val_acc = accuracy_score(all_labels, all_preds) * 100
        print(f"Validation Accuracy: {val_acc:.2f}%")

    return model

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ROOT_DIR = "D:\\Evaluation Set"  # <-- Change to your dataset folder path

    dataset = AudioAttackDataset(ROOT_DIR, max_len=200, sample_rate=16000, n_lfcc=60)
    if len(dataset) == 0:
        raise ValueError("No audio data found in the specified directory.")

    print(f"Total samples: {len(dataset)}, Number of classes: {dataset.num_classes}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    model = train_lcnn_classifier(train_loader, val_loader, DEVICE, epochs=10)
