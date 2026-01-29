import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from tqdm import tqdm
import librosa
import numpy as np
import miditoolkit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
import torchvision.models as models
import pretty_midi
import music21
from collections import Counter
import pandas as pd
import soundfile as sf
import scipy
import joblib
import pickle
from statsmodels.tsa.stattools import acf
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from music21 import spanner
import torchaudio.transforms as T

# 配置和常量
dataroot3 = "student_files/task3_audio_classification/"
TAGS = ['rock', 'oldies', 'jazz', 'pop', 'dance',  'blues',  'punk', 'chill', 'electronic', 'country']
SAMPLE_RATE = 16000
N_MELS = 64
N_CLASSES = 10
AUDIO_DURATION = 10 # seconds
BATCH_SIZE = 4

def accuracy3(groundtruth, predictions):
    preds, targets = [], []
    for k in groundtruth:
        if not (k in predictions):
            print("Missing " + str(k) + " from predictions")
            return 0
        prediction = [1 if tag in predictions[k] else 0 for tag in TAGS]
        target = [1 if tag in groundtruth[k] else 0 for tag in TAGS]
        preds.append(prediction)
        targets.append(target)
    
    mAP = average_precision_score(targets, preds, average='macro')
    return mAP

def extract_waveform(path):
    waveform, sr = librosa.load(dataroot3 + '/' + path, sr=SAMPLE_RATE)
    waveform = np.array([waveform])
    if sr != SAMPLE_RATE:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resample(waveform)
    # Pad so that everything is the right length
    target_len = SAMPLE_RATE * AUDIO_DURATION
    if waveform.shape[1] < target_len:
        pad_len = target_len - waveform.shape[1]
        waveform = F.pad(waveform, (0, pad_len))
    else:
        waveform = waveform[:, :target_len]
    waveform = torch.FloatTensor(waveform)
    return waveform

def augment_waveform(waveform, sample_rate=SAMPLE_RATE):
    # 1. 随机加高斯噪声
    noise_amp = random.uniform(0.001, 0.01)
    noise = noise_amp * torch.randn_like(waveform)
    waveform = waveform + noise

    # 4. 随机静音一段 (Random Silence/Zero Out)
    if random.random() < 0.2:
        length = waveform.shape[-1]
        seg_len = int(length * random.uniform(0.02, 0.05))  # 静音20~50ms
        start = random.randint(0, length - seg_len)
        waveform[..., start:start+seg_len] = 0

    # 5. 随机裁剪并补零 (Random Crop)
    if random.random() < 0.2:
        target_len = int(sample_rate * AUDIO_DURATION)
        max_offset = waveform.shape[-1] - target_len
        if max_offset > 0:
            start = random.randint(0, max_offset)
            waveform = waveform[..., start:start+target_len]
        elif waveform.shape[-1] < target_len:
            pad_len = target_len - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            
    return waveform

def spec_augment(mel_spec, freq_mask_param=8, time_mask_param=16, n_freq_masks=2, n_time_masks=2):
    # 多次mask
    for _ in range(n_freq_masks):
        if random.random() < 0.5:
            mel_spec = T.FrequencyMasking(freq_mask_param)(mel_spec)
    for _ in range(n_time_masks):
        if random.random() < 0.5:
            mel_spec = T.TimeMasking(time_mask_param)(mel_spec)
    # 振幅扰动
    if random.random() < 0.3:
        mel_spec = mel_spec * random.uniform(0.8, 1.2)
    return mel_spec

class AudioDataset(Dataset):
    def __init__(self, meta, preload=True, augment=False,
                 sample_rate=16000, n_mels=64):

        self.meta = meta
        self.paths = list(meta.keys())
        self.preload = preload
        self.augment = augment
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        self.mel = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        self.db = AmplitudeToDB()

        self.pathToFeat = {}           # path → [mel0, mel1, mel2, ...]
        self.flattened_index = []      # 扁平索引: [(path, version_idx)]

        if self.preload:
            print("Preloading mel features (with augmentations)...")
            for path in tqdm(self.paths):
                waveform = extract_waveform(path)
                versions = []

                # 原始 mel
                mel_orig = self.db(self.mel(waveform)).squeeze(0)
                versions.append(mel_orig)

                # 增强后的 mel
                if self.augment:
                    aug_wave = augment_waveform(waveform.clone())
                    mel_waveaug = self.db(self.mel(aug_wave)).squeeze(0)
                    versions.append(mel_waveaug)

                    mel_aug = spec_augment(mel_orig.clone())
                    versions.append(mel_aug)
                    
                self.pathToFeat[path] = versions

            # 构建扁平索引
            for path, mel_list in self.pathToFeat.items():
                for version_idx in range(len(mel_list)):
                    self.flattened_index.append((path, version_idx))

            print(f"Total expanded samples: {len(self.flattened_index)}")

        else:
            self.idToPath = dict(zip(range(len(self.paths)), self.paths))

    def __len__(self):
        if self.preload:
            return len(self.flattened_index)
        else:
            return len(self.meta)

    def __getitem__(self, idx):
        if self.preload:
            path, version_idx = self.flattened_index[idx]
            mel_spec = self.pathToFeat[path][version_idx]
            tags = self.meta[path]
        else:
            path = self.idToPath[idx]
            tags = self.meta[path]
            waveform = extract_waveform(path)
            mel_spec = self.db(self.mel(waveform)).squeeze(0)

        bin_label = torch.tensor([1 if tag in tags else 0 for tag in TAGS], dtype=torch.float32)
        return mel_spec.unsqueeze(0), bin_label, f"{path}#v{version_idx}" if self.preload and self.augment else path

class Loaders():
    def __init__(self, train_path, test_path, split_ratio=0.9, seed = 0):
        torch.manual_seed(seed)
        random.seed(seed)
        
        meta_train = eval(open(train_path, 'r').read())
        l_test = eval(open(test_path, 'r').read())
        meta_test = dict([(x,[]) for x in l_test])
        
        all_train = AudioDataset(meta_train, preload=True, augment=True)
        test_set  = AudioDataset(meta_test, preload=True, augment=False)
        
        # Split all_train into train + valid
        total_len = len(all_train)
        train_len = int(total_len * split_ratio)
        valid_len = total_len - train_len
        train_set, valid_set = random_split(all_train, [train_len, valid_len])
        
        self.loaderTrain = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        self.loaderValid = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        self.loaderTest = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class CNNClassifier(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * (N_MELS // 4) * (801 // 4), 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 16, mel/2, time/2)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 32, mel/4, time/4)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

class CNNRNNClassifier(nn.Module):
    def __init__(self, n_classes, n_mels=64):
        super().__init__()
        # CNN部分
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # RNN部分
        self.rnn_input_dim = (n_mels // 4) * 64
        self.rnn = nn.GRU(input_size=self.rnn_input_dim, hidden_size=128, batch_first=True, bidirectional=True)
        # FC层
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        b, c, h, t = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(b, t, -1)
        x, _ = self.rnn(x)
        x = x.mean(dim=1)
        return self.fc(x)

class Pipeline():
    def __init__(self, model, learning_rate, seed = 0):
        torch.manual_seed(seed)
        random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

    def evaluate(self, loader, threshold=0.5, outpath=None):
        self.model.eval()
        preds, targets, paths = [], [], []
        with torch.no_grad():
            for x, y, ps in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                probs = torch.sigmoid(outputs)
                preds.append(probs.cpu())
                targets.append(y.cpu())
                paths += list(ps)
        
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        preds_bin = (preds > threshold).float()
        
        predictions = {}
        for i in range(preds_bin.shape[0]):
            # Filter out augmentation suffix for saving key
            original_path = paths[i].split('#')[0]
            if original_path not in predictions: # Avoid overwriting if duplicates exist
                predictions[original_path] = [TAGS[j] for j in range(len(preds_bin[i])) if preds_bin[i][j]]
        
        mAP = None
        if outpath: 
            with open(outpath, "w") as z:
                z.write(str(predictions) + '\n')
        else:
            mAP = average_precision_score(targets, preds, average='macro')
        return predictions, mAP

    def train(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for x, y, path in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            val_predictions, mAP = self.evaluate(val_loader)
            print(f"[Epoch {epoch+1}] Loss: {running_loss/len(train_loader):.4f} | Val mAP: {mAP:.4f}")

# --- Main Logic for Task 3 ---
if __name__ == "__main__":
    loaders = Loaders(dataroot3 + "/train.json", dataroot3 + "/test.json")
    
    # 可以在这里切换 CNNClassifier 或 CNNRNNClassifier
    model = CNNClassifier() 
    pipeline = Pipeline(model, 1e-4)

    pipeline.train(loaders.loaderTrain, loaders.loaderValid, 25)
    
    # Evaluation
    train_preds, train_mAP = pipeline.evaluate(loaders.loaderTrain, 0.5)
    valid_preds, valid_mAP = pipeline.evaluate(loaders.loaderValid, 0.5)
    test_preds, _ = pipeline.evaluate(loaders.loaderTest, 0.5, "predictions3.json")
    
    print("Task 3 Done. Saved predictions3.json")