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
from sklearn.utils.class_weight import compute_sample_weight

# 配置路径
dataroot1 = "student_files/task1_composer_classification/"

def extract_features(file_path):
    try:
        pm = pretty_midi.PrettyMIDI(file_path)
        notes = [note for instrument in pm.instruments for note in instrument.notes if not instrument.is_drum]
        if not notes:
            return None

        # --- 音高特征 ---
        pitches = np.array([note.pitch for note in notes])
        pitch_range = pitches.max() - pitches.min()
        pitch_mean = pitches.mean()
        pitch_min = pitches.min()
        pitch_max = pitches.max()
        pitch_mode = np.argmax(np.bincount(pitches))
        pitch_mode_count = np.bincount(pitches)[pitch_mode]
        pitch_mode_ratio = pitch_mode_count / len(pitches)
        pitch_skew = scipy.stats.skew(pitches)
        pitch_hist, _ = np.histogram(pitches, bins=range(21, 109, 12)) 
        low = np.sum(pitches < 50)
        mid = np.sum((pitches >= 50) & (pitches <= 70))

        # --- 节奏特征 ---
        durations = np.array([note.end - note.start for note in notes])
        duration_min = durations.min()
        duration_median = np.median(durations)
        short_ratio = np.sum(durations < 0.25) / len(durations)
        unique, counts = np.unique(durations.round(3), return_counts=True)
        duration_entropy = scipy.stats.entropy(counts)

        # --- 音符密度 ---
        total_time = pm.get_end_time()
        note_density = len(notes) / total_time if total_time > 0 else 0
        onset_times = np.array([note.start for note in notes])

        onsets_sorted = np.sort(onset_times)
        inter_onset_intervals = np.diff(onsets_sorted)
        legato_ratio = np.sum(inter_onset_intervals < 0.05) / len(inter_onset_intervals)
        
        start_times = np.array([note.start for note in notes])
        intervals = np.diff(np.sort(start_times))

        # --- 力度特征 ---
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
        weak_ratio = np.sum(velocities < 60) / len(velocities)

        # --- 多音性 ---
        polyphony_window = np.array([
            np.sum((onset_times >= t) & (onset_times < t + 0.1))
            for t in np.arange(0, total_time, 0.1)
        ])
        min_polyphony = polyphony_window.min()

        # --------------------
        # 使用 music21 分析高级特征
        # --------------------
        score = music21.converter.parse(file_path)

        # 节拍（拍号）
        try:
            ts = score.recurse().getElementsByClass('TimeSignature')[0]
            time_signature = ts.ratioString
            numerator, denominator = map(int, time_signature.split('/'))
        except:
            time_signature = 'Unknown'
            numerator, denominator = 0, 0

        # 拍点强度
        beat_strengths = [n.beatStrength for n in score.flat.notes if hasattr(n, 'beatStrength')]
        beat_strength_mean = np.mean(beat_strengths) if beat_strengths else 0

        # 音程
        notes21 = []
        for n in score.flat.notes:
            if isinstance(n, music21.note.Note):
                notes21.append(n)
            elif isinstance(n, music21.chord.Chord):
                notes21.append(n.notes[0]) 
        intervals = [music21.interval.Interval(notes21[i], notes21[i+1]) for i in range(len(notes21)-1)]
        signed = np.array([iv.semitones for iv in intervals])
        absed = np.abs(signed)

        # 和弦熵
        chords = score.chordify().flat.getElementsByClass('Chord')
        chord_names = [c.pitchedCommonName for c in chords if c.isTriad()]
        chord_entropy = scipy.stats.entropy(list(Counter(chord_names).values()), base=2)

        # 休止比例
        rests = score.flat.getElementsByClass('Rest')
        rest_duration = sum(r.duration.quarterLength for r in rests)
        total_duration = score.duration.quarterLength
        rest_ratio = rest_duration / total_duration if total_duration > 0 else 0

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
            'weak_velocity_ratio': weak_ratio,
            'min_polyphony' : min_polyphony,
            "intv_signed_mean": signed.mean(),
            "intv_signed_std": signed.std(),
            "intv_signed_min": signed.min(),
            "intv_signed_max": signed.max(),
            "intv_abs_mean": absed.mean(),
            "intv_abs_std": absed.std(),
            "intv_abs_min": absed.min(),
            "intv_abs_max": absed.max(),
            "intv_pos_ratio": (signed > 0).mean(),
            "intv_neg_ratio": (signed < 0).mean(),
            "intv_zero_ratio": (signed == 0).mean(),
            'chord_entropy': chord_entropy,
            'rest_ratio': rest_ratio,
            'beat_strength_mean': beat_strength_mean,
            'numerator' : numerator,
            'denominator' : denominator,
        }
            # --- Pedal 特征 ---
        pedals21 = [obj for obj in score.spanners if isinstance(obj, spanner.SustainPedal)]
        if pedals21:
            durations = [pedal.getDuration() for pedal in pedals21 if pedal.getDuration() is not None]
            
            if durations:
                durations = np.array(durations)
                features.update({
                    "pedal_count": len(durations),
                    "pedal_mean_duration": durations.mean(),
                    "pedal_max_duration": durations.max(),
                    "pedal_total_duration": durations.sum(),
                })
            else:
                features.update({
                    "pedal_count": len(pedals21),
                    "pedal_mean_duration": 0,
                    "pedal_max_duration": 0,
                    "pedal_total_duration": 0,
                })
        else:
            features.update({
                "pedal_count": 0,
                "pedal_mean_duration": 0,
                "pedal_max_duration": 0,
                "pedal_total_duration": 0,
            })

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- Main Logic for Task 1 ---
if __name__ == "__main__":
    # 1. Loading Training Data
    if os.path.exists("f1.pkl"):
        print("Loading cached features from f1.pkl...")
        X_read, y_read = joblib.load("f1.pkl")
    else:
        print("Extracting features (this may take a while)...")
        d = eval(open(dataroot1 + "/train.json", 'r').read())
        X = []
        y = []

        for k in tqdm(d):
            try:
                feat = extract_features(dataroot1 + '/' + k)
                X.append(feat)
                y.append(d[k])
            except Exception as e:
                print(f"Error on {k}: {e}")

        X_read = pd.DataFrame(X)
        y_read = pd.Series(y)
        joblib.dump((X_read, y_read), "f1.pkl")

    # 2. Preprocessing
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_read)
    
    # 转换失败设为 NaN 并补 0
    X_read = X_read.apply(pd.to_numeric, errors='coerce') 
    X_read = X_read.fillna(0)

    scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X_read) # Optional if needed

    # 3. Training
    print("Training Model...")
    weights = compute_sample_weight(class_weight='balanced', y=y_encoded)
    
    clf = LGBMClassifier(
        objective='multiclass',
        num_class=len(le.classes_),
        num_leaves=2000,
        learning_rate=0.01,
        n_estimators=2000,
        reg_alpha=0.05,
        reg_lambda=0.1,
    )

    clf.fit(X_read, y_encoded, sample_weight=weights)

    # 4. Feature Importance (Optional)
    # import matplotlib.pyplot as plt
    # from xgboost import plot_importance
    # fig, ax = plt.subplots(figsize=(8, 20))
    # plot_importance(clf, importance_type='gain', ax=ax)
    # plt.show()

    # 5. Prediction
    print("Processing Test Data...")
    if os.path.exists("f2.pkl"):
        X_test_1 = joblib.load("f2.pkl")
    else:
        d_test = eval(open(dataroot1 + "/test.json", 'r').read())
        X_test_1_list = []
        for k in tqdm(d_test):
            try:
                feat = extract_features(dataroot1 + '/' + k)
                X_test_1_list.append(feat)
            except Exception as e:
                print(f"Error on {k}: {e}")
        X_test_1 = pd.DataFrame(X_test_1_list)
        joblib.dump((X_test_1), "f2.pkl")

    X_test_1 = X_test_1.apply(pd.to_numeric, errors='coerce')
    X_test_1 = X_test_1.fillna(0)
    
    # Get Keys for mapping
    d_test = eval(open(dataroot1 + "/test.json", 'r').read())
    keys = []
    for k in d_test:
        keys.append(k)

    print("Predicting...")
    preds = clf.predict(X_test_1)
    decoded_preds = le.inverse_transform(preds.astype(int).ravel())
    
    predictions = {keys[i]: str(decoded_preds[i]) for i in range(len(keys))}

    with open('predictions1.json', "w") as z:
        z.write(str(predictions) + '\n')
    print("Task 1 Done. Saved predictions1.json")