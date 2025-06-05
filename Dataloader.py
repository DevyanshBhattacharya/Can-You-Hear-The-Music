import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

# Model input configurations
MODEL_INPUT_CONFIG = {
    'SimpleYAMNet': {'type': 'mel', 'shape': (1, 96, 64)},  # Expected [B, 1, 96, 64]
    'RawNet': {'type': 'raw', 'shape': (1, 64000)},         # Expected [B, 1, 64000]
    'LFCC_LCNN': {'type': 'lfcc', 'shape': (1, 100, 60)},   # Expected [B, 1, 100, 60]
    'LFCC_GMR': {'type': 'lfcc_flat', 'shape': (None, 60)}  # Expected [N, 60] (flattened)
}

class AudioAttackDataset(Dataset):
    def __init__(self, root_dir, model_name='SimpleYAMNet', max_length=64000):
        self.samples = []
        self.labels = []
        self.model_name = model_name
        self.max_length = max_length
        self.label_map = {}

        if model_name not in MODEL_INPUT_CONFIG:
            raise ValueError(f"Unknown model_name '{model_name}'. Supported: {list(MODEL_INPUT_CONFIG.keys())}")

        self.feature_type = MODEL_INPUT_CONFIG[model_name]['type']
        subfolders = sorted(os.listdir(root_dir))
        for idx, folder in enumerate(subfolders):
            self.label_map[folder] = idx
            folder_path = os.path.join(root_dir, folder)
            for file in os.listdir(folder_path):
                if file.lower().endswith((".flac", ".wav", ".mp3")):
                    self.samples.append(os.path.join(folder_path, file))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]
        waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono

        if self.feature_type == 'mel':
            mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=64)(waveform)
            db = torchaudio.transforms.AmplitudeToDB()(mel)  # [1, 64, T]
            db = F.interpolate(db.unsqueeze(0), size=(96, 64), mode='bilinear', align_corners=False).squeeze(0)
            return db, label
        elif self.feature_type == 'raw':
            if waveform.shape[1] < self.max_length:
                waveform = F.pad(waveform, (0, self.max_length - waveform.shape[1]))
            else:
                waveform = waveform[:, :self.max_length]
            return waveform, label
        else:
            raise ValueError(f"Unsupported feature_type: {self.feature_type}")
