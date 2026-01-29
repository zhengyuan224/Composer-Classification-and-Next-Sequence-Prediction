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

# 配置路径
dataroot2 = "student_files/task2_next_sequence_prediction/"

def extract_features(file_path):
    try:
        pm = pretty_midi.PrettyMIDI(file_path)
        notes = [note for instrument in pm.instruments for note in instrument.notes if not instrument.is_drum]
        if not notes:
            return None

        # 音高特征
        pitches = np.array([note.pitch for note in notes])
        pitch_range = pitches.max() - pitches.min()
        pitch_mean = pitches.mean()
        pitch_min = pitches.min()
        pitch_max = pitches.max()
        # pitch_mode = np.argmax(np.bincount(pitches))
        pitch_mode_count = np.bincount(pitches)[np.argmax(np.bincount(pitches))]
        pitch_mode_ratio = pitch_mode_count / len(pitches)
        pitch_skew = scipy.stats.skew(pitches)
        
        low = np.sum(pitches < 50)
        mid = np.sum((pitches >= 50) & (pitches <= 70))

        # 节奏特征
        durations = np.array([note.end - note.start for note in notes])
        duration_min = durations.min()
        duration_median = np.median(durations)
        short_ratio = np.sum(durations < 0.25) / len(durations)
        unique, counts = np.unique(durations.round(3), return_counts=True)
        duration_entropy = scipy.stats.entropy(counts)

        # 音符密度
        total_time = pm.get_end_time()
        note_density = len(notes) / total_time if total_time > 0 else 0
        onset_times = np.array([note.start for note in notes])

        onsets_sorted = np.sort(onset_times)
        inter_onset_intervals = np.diff(onsets_sorted)
        legato_ratio = np.sum(inter_onset_intervals < 0.05) / len(inter_onset_intervals)
        
        # 力度特征
        velocities = np.array([note.velocity for note in notes])
        velocity_mean = velocities.mean()
        velocity_std = velocities.std()
        velocity_min = velocities.min()
        velocity_max = velocities.max()
        velocity_median = np.median(velocities)
        velocity_change = np.abs(np.diff(velocities))
        velocity_change_mean = velocity_change.mean()
        velocity_change_std = velocity_change.std()
        velocity_change_max = velocity_change.max()
        velocity_change_min = velocity_change.min()
        velocity_change_median = np.median(velocity_change)
        velocity_change_entropy = scipy.stats.entropy(velocity_change)
        velocity_change_range = velocity_change.max() - velocity_change.min()

        features = {
            'pitch_range': pitch_range,
            'pitch_mean': pitch_mean,
            'pitch_min': pitch_min,
            'pitch_max': pitch_max,
            'pitch_mode_ratio': pitch_mode_ratio,
            'pitch_skew': pitch_skew,
            'low_pitch_ratio': low / len(pitches),
            'mid_pitch_ratio': mid / len(pitches),
            'duration_min': duration_min,
            'duration_median': duration_median,
            'short_duration_ratio': short_ratio,
            'duration_entropy': duration_entropy,
            'note_density': note_density,
            'legato_ratio': legato_ratio,
            'velocity_mean': velocity_mean,
            'velocity_std': velocity_std,
            'velocity_median': velocity_median,
            'velocity_min': velocity_min,
            'velocity_max': velocity_max,
            'velocity_change_mean': velocity_change_mean,
            'velocity_change_std': velocity_change_std,
            'velocity_change_median': velocity_change_median,
            'velocity_change_min': velocity_change_min,
            'velocity_change_max': velocity_change_max,
            'velocity_change_entropy': velocity_change_entropy,
            'velocity_change_range': velocity_change_range,
        }
        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- Main Logic for Task 2 ---
if __name__ == "__main__":
    # 1. Loading and Processing Train Data
    print("Processing Train Data...")
    d = eval(open(dataroot2 + "/train.json", 'r').read())
    X, y = [], []

    for k in tqdm(d):
        path1, path2 = k
        x1 = extract_features(dataroot2 + path1)
        x2 = extract_features(dataroot2 + path2)

        # Calculate difference between features (Siamese-like approach)
        if x1 and x2:
            feature_diff = [abs(a - b) for a, b in zip(x1.values(), x2.values())]
            X.append(feature_diff)
            y.append(int(d[k]))

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  

    # 2. Validation (Optional Step in script, good for checking performance)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_y_true = []
    all_y_pred = []
    
    model = LGBMClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        objective='binary',
        random_state=42,
        reg_alpha=0.1,
        reg_lambda=1,
        feature_fraction = 0.8
    )

    print("Running Cross Validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_encoded)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        # print(f"Fold {fold + 1} Accuracy: {acc:.4f}")
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
    
    print("Overall CV Accuracy:", accuracy_score(all_y_true, all_y_pred))

    # 3. Retrain on full data and Predict on Test
    print("Processing Test Data...")
    d_test = eval(open(dataroot2 + "/test.json", 'r').read())
    X_test_list, keys = [], []

    for k in tqdm(d_test):
        path1, path2 = k
        x1 = extract_features(dataroot2 + path1)
        x2 = extract_features(dataroot2 + path2)
        
        if x1 and x2:
            feature_diff = [abs(a - b) for a, b in zip(x1.values(), x2.values())]
            X_test_list.append(feature_diff)
            keys.append(k)
        else:
            # Handle error case (should ideally not happen if data is clean)
            X_test_list.append([0]*X.shape[1]) 
            keys.append(k)

    X_test = np.array(X_test_list)
    X_test_scaled = scaler.transform(X_test)

    print("Refitting on full train data and predicting...")
    model.fit(X_scaled, y_encoded)
    preds = model.predict(X_test_scaled)
    decoded_preds = le.inverse_transform(preds.astype(int).ravel())

    predictions = {str(keys[i]): bool(decoded_preds[i]) for i in range(len(keys))}

    with open('predictions2.json', "w") as z:
        z.write(str(predictions) + '\n')
    print("Task 2 Done. Saved predictions2.json")