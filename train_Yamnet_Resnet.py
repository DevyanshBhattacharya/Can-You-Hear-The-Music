from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

# ──────── Hyper-parameters ────────────────────────────────────────────────
ROOT_DIR        = r"D:\Evaluation Set"   # ← change to your dataset root
EPOCHS          = 30
BATCH_SIZE      = 16
LR              = 1e-4
PATIENCE        = 5
SNAPSHOT_INT    = 10
SAVE_DIR        = Path("models_yamnet_rawnet")
SAMPLE_RATE     = 16_000
TARGET_LEN_RAW  = 16_000   # RawNet (1 s)
TARGET_LEN_YAM  = 64_000   # YAMNet (~4 s)

np.random.seed(42)
torch.manual_seed(42)

# ──────── Model definitions (replace with real) ───────────────────────────
class SimpleYAMNet(nn.Module):
    def __init__(self, num_classes:int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(128, num_classes)
    def forward(self,x):
        return self.fc(self.conv(x).squeeze(-1))

class RawNet(nn.Module):
    def __init__(self, num_classes:int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,64,5,stride=3,padding=2), nn.ReLU(), nn.MaxPool1d(3),
            nn.Conv1d(64,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(128, num_classes)
    def forward(self,x):
        return self.fc(self.conv(x).squeeze(-1))

# ──────── Dataset ─────────────────────────────────────────────────────────
class AudioAttackDataset(Dataset):
    def __init__(self, root:str|Path, model_name:str):
        self.root      = Path(root)
        self.model_name= model_name
        self.label_map = {d.name:i for i,d in enumerate(sorted(self.root.iterdir()) ) if d.is_dir()}
        self.files, self.labels = [], []
        for lbl in self.label_map:
            for f in (self.root/lbl).glob('*.flac'): self.files.append(f); self.labels.append(self.label_map[lbl])
        print(f"[{model_name}] loaded {len(self.files)} files across {len(self.label_map)} classes")

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.files[idx])
        if sr!=SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.mean(dim=0, keepdim=True)  # mono
        target = TARGET_LEN_RAW if self.model_name=='RawNet' else TARGET_LEN_YAM
        wav = _fix_len(wav,target)
        return wav, self.labels[idx]

def _fix_len(w, L):
    return F.pad(w,(0,max(0,L-w.shape[1])))[:,:L]

# ──────── helpers ─────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR.mkdir(exist_ok=True)

def save_cm(cm:np.ndarray, epoch:int, names, out:Path):
    out.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm,annot=True,fmt='d',cbar=False,xticklabels=names,yticklabels=names)
    plt.tight_layout(); plt.savefig(out/f'cm_{epoch:03d}.png'); plt.close()

def run_epoch(model,dl,crit,opt=None):
    model.train() if opt else model.eval()
    y_pred,y_true,loss= [],[],0.0
    for x,y in dl:
        x,y=x.to(DEVICE),y.to(DEVICE)
        if opt: opt.zero_grad()
        with torch.set_grad_enabled(opt is not None):
            out=model(x); l=crit(out,y)
            if opt: l.backward(); opt.step()
        loss+=l.item()*x.size(0)
        y_pred+=out.argmax(1).cpu().tolist(); y_true+=y.cpu().tolist()
    loss/=len(dl.dataset)
    acc=accuracy_score(y_true,y_pred)
    prec,rec,f1,_=precision_recall_fscore_support(y_true,y_pred,average='macro',zero_division=0)
    cm=confusion_matrix(y_true,y_pred)
    return loss,acc,prec,rec,f1,cm

# ──────── train wrapper ───────────────────────────────────────────────────
def train_model(name:str, model_cls):
    ds=AudioAttackDataset(ROOT_DIR,name)
    tr_len=int(0.8*len(ds)); tr_ds,val_ds=random_split(ds,[tr_len,len(ds)-tr_len])
    tr_dl=DataLoader(tr_ds,BATCH_SIZE,shuffle=True); val_dl=DataLoader(val_ds,BATCH_SIZE)

    model=model_cls(len(ds.label_map)).to(DEVICE)
    crit=nn.CrossEntropyLoss(); opt=optim.Adam(model.parameters(),lr=LR)
    best_f1=-1; stall=0
    best_path=SAVE_DIR/f'{name}_best.pt'
    cm_dir=SAVE_DIR/'cms'; snap_dir=SAVE_DIR/'snapshots'
    for ep in range(1,EPOCHS+1):
        tr_loss,_,_,_,_,_ = run_epoch(model,tr_dl,crit,opt)
        vl_loss,vl_acc,vl_prec,vl_rec,vl_f1,cm = run_epoch(model,val_dl,crit)
        print(f"{name} Ep{ep:02d} loss {vl_loss:.4f} f1 {vl_f1:.3f} acc {vl_acc:.3f}")
        save_cm(cm,ep,list(ds.label_map),cm_dir)
        if ep%SNAPSHOT_INT==0:
            torch.save(model.state_dict(),snap_dir/f'{name}_ep{ep:03d}.pt')
        if vl_f1>best_f1:
            best_f1=vl_f1; stall=0; torch.save(model.state_dict(),best_path); print('  ↳ best saved')
        else:
            stall+=1
            if stall>=PATIENCE:
                print(f'Early stopping after {stall} stalls')
                break

    model.load_state_dict(torch.load(best_path)); _,_,_,_,_,cm=run_epoch(model,val_dl,crit)
    y_pred,y_true=[],[]
    with torch.no_grad():
        for x,y in val_dl:
            y_true+=y.tolist(); y_pred+=model(x.to(DEVICE)).argmax(1).cpu().tolist()
    print(classification_report(y_true,y_pred,target_names=list(ds.label_map)))

# ──────── main ─────────────────────────────────────────────────────────────
for n,cls in [('SimpleYAMNet',SimpleYAMNet),('RawNet',RawNet)]:
    print(f"\n===== {n} ====="); train_model(n,cls)
