# %%
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

# %%
dataroot1 = "student_files/task1_composer_classification/"

# %%
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
        # pitch_std = pitches.std()
        #new
        # pitch_median = np.median(pitches)
        pitch_min = pitches.min()
        pitch_max = pitches.max()
        pitch_mode = np.argmax(np.bincount(pitches))
        pitch_mode_count = np.bincount(pitches)[pitch_mode]
        pitch_mode_ratio = pitch_mode_count / len(pitches)
        pitch_skew = scipy.stats.skew(pitches)
        # pitch_kurtosis = scipy.stats.kurtosis(pitches)
        pitch_hist, _ = np.histogram(pitches, bins=range(21, 109, 12))  # 按八度划分
        # note_repeats = np.sum(np.diff(pitches) == 0)
        # repeat_ratio = note_repeats / len(pitches)
        low = np.sum(pitches < 50)
        mid = np.sum((pitches >= 50) & (pitches <= 70))
        # high = np.sum(pitches > 70)

        # pitch_counts = Counter(pitches)
        # pitch_probs = np.array(list(pitch_counts.values())) / len(pitches)
        # if pitch_probs.sum() > 0:
        #     pitch_entropy = scipy.stats.entropy(pitch_probs, base=2)
        # else:
        #     pitch_entropy = 0

        # 节奏特征
        durations = np.array([note.end - note.start for note in notes])
        # duration_mean = durations.mean()
        # duration_std = durations.std()
        #new
        duration_min = durations.min()
        # duration_max = durations.max()
        duration_median = np.median(durations)
        # duration_skew = scipy.stats.skew(durations)
        # duration_kurtosis = scipy.stats.kurtosis(durations)
        # duration_change = np.abs(np.diff(durations))
        # duration_change_mean = duration_change.mean()
        # duration_change_std = duration_change.std()
        short_ratio = np.sum(durations < 0.25) / len(durations)
        # long_ratio = np.sum(durations > 1.0) / len(durations)
        unique, counts = np.unique(durations.round(3), return_counts=True)
        duration_entropy = scipy.stats.entropy(counts)

        # rhythm_entropy = scipy.stats.entropy(np.bincount((durations * 4).astype(int)))  # 用近似四分音符单位的时值编码
        # acf_vals = acf(durations, nlags=10, fft=True)


        # 音符密度
        total_time = pm.get_end_time()
        note_density = len(notes) / total_time if total_time > 0 else 0
        #new
        # step = 1.0  # 1秒为一个时间段
        onset_times = np.array([note.start for note in notes])
        # segments = np.arange(0, total_time, step)
        # densities = [np.sum((onset_times >= t) & (onset_times < t + step)) for t in segments]
        # density_std = np.std(densities)
        # density_max = np.max(densities)
        # density_min = np.min(densities)
        # density_range = density_max - density_min


        onsets_sorted = np.sort(onset_times)
        inter_onset_intervals = np.diff(onsets_sorted)
        legato_ratio = np.sum(inter_onset_intervals < 0.05) / len(inter_onset_intervals)
        
        start_times = np.array([note.start for note in notes])
        intervals = np.diff(np.sort(start_times))

        # 力度特征
        velocities = np.array([note.velocity for note in notes])
        velocity_mean = velocities.mean()
        velocity_std = velocities.std()
        #new
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
        # velocity_skew = scipy.stats.skew(velocities)
        # velocity_kurtosis = scipy.stats.kurtosis(velocities)
        # strong_ratio = np.sum(velocities > 100) / len(velocities)
        weak_ratio = np.sum(velocities < 60) / len(velocities)


        # 多音性
        polyphony_window = np.array([
            np.sum((onset_times >= t) & (onset_times < t + 0.1))
            for t in np.arange(0, total_time, 0.1)
        ])
        # polyphony = polyphony_window.mean()
        # max_polyphony = polyphony_window.max()
        min_polyphony = polyphony_window.min()
        # polyphony_std = polyphony_window.std()




        # --------------------
        # 使用 music21 分析高级特征
        # --------------------
        score = music21.converter.parse(file_path)

        # 调性分析
        # key_obj = score.analyze('key')
        # tonic = key_obj.tonic.name
        # mode = key_obj.mode
        # key_confidence = key_obj.correlationCoefficient
        # mode_map = {'major': 1, 'minor': 0}
        # tonic_map = {
        #     'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
        #     'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
        # }
        # mode_num = mode_map.get(mode, -1)
        # tonic_num = tonic_map.get(tonic, -1)

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
                notes21.append(n.notes[0])  # 或 n.notes[-1], n.root(), n.pitches 等看你需要
        intervals = [music21.interval.Interval(notes21[i], notes21[i+1]) for i in range(len(notes21)-1)]
        signed = np.array([iv.semitones for iv in intervals])
        absed = np.abs(signed)


        # 和弦熵
        chords = score.chordify().flat.getElementsByClass('Chord')
        chord_names = [c.pitchedCommonName for c in chords if c.isTriad()]
        chord_entropy = scipy.stats.entropy(list(Counter(chord_names).values()), base=2)

        # avg_chord_len = np.mean([c.quarterLength for c in chords]) if chords else 0
        # chord_variety = len(set(chord_names))

        # 小节结构
        # try:
        #     measures = score.parts[0].getElementsByClass('Measure')
        #     note_counts_per_measure = [len(m.notes) for m in measures]
            # note_density_var = np.std(note_counts_per_measure)
            # phrase_lengths = [len(m.notes) for m in measures] if measures else []
            # phrase_std = np.std(phrase_lengths) if phrase_lengths else 0
        # except:
        #     note_density_var = 0.0

        # 休止比例
        rests = score.flat.getElementsByClass('Rest')
        rest_duration = sum(r.duration.quarterLength for r in rests)
        total_duration = score.duration.quarterLength
        rest_ratio = rest_duration / total_duration if total_duration > 0 else 0

        # offbeats = [n.beat % 1 > 0.4 for n in score.flat.notes]
        # offbeat_ratio = np.sum(offbeats) / len(offbeats) if offbeats else 0


        features = {
            'pitch_range': pitch_range,
            'pitch_mean': pitch_mean,
            # 'pitch_std': pitch_std,

            # 'pitch_median': pitch_median,
            'pitch_min': pitch_min,
            'pitch_max': pitch_max,
            # 'pitch_mode': pitch_mode,
            # 'pitch_mode_count': pitch_mode_count,
            'pitch_mode_ratio': pitch_mode_ratio,
            'pitch_skew': pitch_skew,
            # 'pitch_kurtosis': pitch_kurtosis,
            # 'repeat_ratio': repeat_ratio,
            'low_pitch_ratio': low / len(pitches),
            'mid_pitch_ratio': mid / len(pitches),
            # 'high_pitch_ratio': high / len(pitches),

            # 'pitch_entropy': pitch_entropy,

            # 'duration_mean': duration_mean,
            # 'duration_std': duration_std,

            'duration_min': duration_min,
            # 'duration_max': duration_max,
            'duration_median': duration_median,
            # 'duration_skew': duration_skew,
            # 'duration_kurtosis': duration_kurtosis,
            # 'duration_change_mean': duration_change_mean,
            # 'duration_change_std': duration_change_std,
            'short_duration_ratio': short_ratio,
            # 'long_duration_ratio': long_ratio,
            'duration_entropy': duration_entropy,
            
            # 'rhythm_entropy': rhythm_entropy,
            # 'duration_acf1': acf_vals[1],

            'note_density': note_density,
            # 'density_std' : density_std,
            # 'density_max' : density_max,
            # 'density_min' : density_min,
            # 'density_range' : density_range,

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
            # 'velocity_skew': velocity_skew,
            # 'velocity_kurtosis': velocity_kurtosis,
            # 'strong_velocity_ratio': strong_ratio,
            'weak_velocity_ratio': weak_ratio,

            # 'polyphony': polyphony,

            # 'max_polyphony' : max_polyphony,
            'min_polyphony' : min_polyphony,
            # 'polyphony_std' : polyphony_std,

            # 'mode_num': mode_num,
            # 'tonic_num': tonic_num,
            # 'key_confidence': key_confidence,

            # Music21 additional
            # 'interval_variety': interval_variety,
            # 有符号音程特征（表示走向）
            "intv_signed_mean": signed.mean(),
            "intv_signed_std": signed.std(),
            "intv_signed_min": signed.min(),
            "intv_signed_max": signed.max(),

            # 绝对值音程特征（表示跳跃幅度）
            "intv_abs_mean": absed.mean(),
            "intv_abs_std": absed.std(),
            "intv_abs_min": absed.min(),
            "intv_abs_max": absed.max(),

            # 上行、下行、无变化比例
            "intv_pos_ratio": (signed > 0).mean(),
            "intv_neg_ratio": (signed < 0).mean(),
            "intv_zero_ratio": (signed == 0).mean(),
            'chord_entropy': chord_entropy,

            # 'avg_chord_len': avg_chord_len,
            # 'chord_variety': chord_variety,
            
            # 'note_density_var': note_density_var,
            # 'phrase_std': phrase_std,
            
            'rest_ratio': rest_ratio,
            'beat_strength_mean': beat_strength_mean,
            'numerator' : numerator,
            'denominator' : denominator,

            # 'offbeat_ratio': offbeat_ratio,

        }
            # --- Pedal 特征 ---
        pedals21 = [obj for obj in score.spanners if isinstance(obj, spanner.SustainPedal)]
        if pedals21:
            durations = [pedal.getDuration() for pedal in pedals21 if pedal.getDuration() is not None]
            starts = [pedal.offset for pedal in pedals21]

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

        # for i, val in enumerate(pitch_hist):
        #     features[f'pitch_hist_octave_{i}'] = val

        # pitch_classes = [note.pitch % 12 for note in notes]
        # pitch_classes = np.array(pitch_classes)
        # pitch_class_hist = np.bincount(pitch_classes, minlength=12)
        # for i in range(12):
        #     features[f'pc_{i}'] = pitch_class_hist[i] / np.sum(pitch_class_hist)
        # features['pitch_class_entropy'] = scipy.stats.entropy(pitch_class_hist / np.sum(pitch_class_hist))

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# %%
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


X = pd.DataFrame(X)
y = pd.Series(y)


# %%
joblib.dump((X, y), "f1.pkl")

# %%
X_read, y_read = joblib.load("f1.pkl")
le = LabelEncoder()
y_encoded = le.fit_transform(y_read)  

# %%
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight


# %%
columns_to_remove = ['pitch_class_entropy', 'pc_0', 'pc_3', 'avg_chord_len', 'pc_10', 'pc_11', 'pitch_hist_octave_5', 'pc_4', 'pitch_kurtosis', 'pc_5', 'pitch_hist_octave_2', 'duration_kurtosis', 'polyphony_std', 'pitch_std', 'velocity_skew', 'duration_acf1', 'pc_6', 'polyphony', 'interval_variety', 'duration_max', 'pitch_hist_octave_1', 'note_density_var', 'pitch_mode_count', 'chord_variety', 'offbeat_ratio', 'pitch_entropy', 'high_pitch_ratio', 'long_duration_ratio', 'density_std', 'duration_std', 'denominator', 'density_min', 'key_confidence', 'pitch_hist_octave_4', 'density_range', 'duration_change_mean', 'repeat_ratio', 'duration_skew', 'strong_velocity_ratio', 'density_max', 'rhythm_entropy', 'pitch_median', 'duration_mean', 'duration_change_std', 'pitch_mode', 'tonic_num', 'phrase_std', 'velocity_kurtosis', 'pitch_hist_octave_0', 'max_polyphony', 'mode_num']
def drop_columns_by_name(df: pd.DataFrame, columns_to_drop):
    if isinstance(columns_to_drop, str):
        columns_to_drop = [columns_to_drop]

    # 过滤出存在的列
    valid_cols = [col for col in columns_to_drop if col in df.columns]

    # 删除这些列
    return df.drop(columns=valid_cols)

# %%
X_filtered = drop_columns_by_name(X_read, columns_to_remove)

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_read)


# %%
X_filtered = X_filtered.apply(pd.to_numeric, errors='coerce')  # 转换失败设为 NaN
X_filtered = X_filtered.fillna(0)  # 再补 0

# %%
X_read = X_read.apply(pd.to_numeric, errors='coerce')  # 转换失败设为 NaN
X_read = X_read.fillna(0)  # 再补 0

# %%
# 创建 KFold
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

accuracies = []

for train_idx, val_idx in cv.split(X_read, y_encoded):
    X_train, X_val = X_read.iloc[train_idx], X_read.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    # 计算 sample_weight
    weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # clf = XGBClassifier(
    #     objective='multi:softprob',
    #     num_class=len(le.classes_),
    #     max_depth=6,
    #     learning_rate=0.1,
    #     n_estimators=1000,
    #     eval_metric='mlogloss',
    #     reg_alpha=0,
    #     reg_lambda=1,
    #     colsample_bytree= 1,
    #     subsample=0.8,
    #     random_state=42
    # )

#     clf = LGBMClassifier(
#     objective='multiclass',
#     num_class=len(le.classes_),
#     max_depth=6,
#     learning_rate=0.01,
#     n_estimators=1000,
#     reg_alpha=0.01,
#     reg_lambda=0.01,
#     colsample_bytree=1.0,
#     subsample=0.8,
#     random_state=42
# )
    clf = LGBMClassifier(
    objective='multiclass',
    num_class=len(le.classes_),
    num_leaves=100,
    learning_rate=0.01,
    n_estimators=1000,
)

    # 拟合模型，加入样本权重
    clf.fit(X_train, y_train, sample_weight=weights)

    # 预测 & 评估
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    accuracies.append(acc)

# 输出最终结果
print("每一折准确率：", accuracies)
print("平均准确率：", np.mean(accuracies))

# %%
import matplotlib.pyplot as plt
from xgboost import plot_importance

fig, ax = plt.subplots(figsize=(8, 20))
clf.get_booster().feature_names = list(X_read.columns)
plot_importance(clf, importance_type='gain', ax=ax)  # 可选：'weight', 'gain', 'cover'
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# %%
importance_dict = clf.get_booster().get_score(importance_type='total_gain')

# 排序并提取特征名
sorted_features = [k for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)]

# 输出结果
print(sorted_features)


# %%
from catboost import CatBoostClassifier

# 创建 KFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

for train_idx, val_idx in cv.split(X_scaled, y_encoded):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    # 计算 sample_weight
    weights = compute_sample_weight(class_weight='balanced', y=y_train)

    clf = CatBoostClassifier(
        # 基础参数
        loss_function='MultiClassOneVsAll',
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        random_seed=42,

        # 正则化和随机性控制
        l2_leaf_reg=3,
        random_strength=1.0,   # 分裂随机性控制
        # rsm=0.8,  # 随机特征子集

        # 💻 硬件加速
        task_type='GPU',    # 或设为 'GPU' 用于 GPU 训练
        devices='0',        # GPU 编号（如果使用 GPU）

        # 提前停止
        early_stopping_rounds=50,
        use_best_model=True,

        # 模型评估
        eval_metric='Accuracy',

        # 输出控制
        verbose=0
        
    )

    # 拟合模型，加入样本权重
    clf.fit(
        X_train, y_train,
        sample_weight=weights,                # 加权训练（可选）
        eval_set=(X_val, y_val),              # 验证集（用于 early stopping）
        verbose=100
    )

    # 预测 & 评估
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    accuracies.append(acc)

# 输出最终结果
print("每一折准确率：", accuracies)
print("平均准确率：", np.mean(accuracies))

# %%
d = eval(open(dataroot1 + "/test.json", 'r').read())

X_test_1 = []
keys = []

for k in tqdm(d):
    try:
        feat = extract_features(dataroot1 + '/' + k)
        X_test_1.append(feat)
        keys.append(k)  # 保存文件名顺序
    except Exception as e:
        print(f"Error on {k}: {e}")

# 转为 NumPy 并归一化
X_test_1 = pd.DataFrame(X_test_1)


# %%
joblib.dump((X_test_1), "f2.pkl")

# %%
X_test_1 = joblib.load("f2.pkl")

X_test_1 = X_test_1.apply(pd.to_numeric, errors='coerce')  # 转换失败设为 NaN
X_test_1 = X_test_1.fillna(0)  # 再补 0

# X_filtered_test = drop_columns_by_name(X_test_1, columns_to_remove)


# %%
scaled_features_test = scaler.transform(X_test_1)

# %%
d = eval(open(dataroot1 + "/test.json", 'r').read())
keys = []
for k in tqdm(d):
    try:
        keys.append(k)  # 保存文件名顺序
    except Exception as e:
        print(f"Error on {k}: {e}")

# %%
# clf = CatBoostClassifier(
#     # 基础参数
#     loss_function='MultiClassOneVsAll',
#     iterations=500,
#     learning_rate=0.1,
#     depth=6,
#     random_seed=42,

#     # 正则化和随机性控制
#     l2_leaf_reg=3,
#     random_strength=1.0,   # 分裂随机性控制
#     # rsm=0.8,  # 随机特征子集

#     # 💻 硬件加速
#     task_type='GPU',    # 或设为 'GPU' 用于 GPU 训练
#     devices='0',        # GPU 编号（如果使用 GPU）

#     # 提前停止
#     early_stopping_rounds=100,
#     # use_best_model=True,

#     # 模型评估
#     eval_metric='Accuracy',

#     # 输出控制
#     verbose=0
    
# )
# weights = compute_sample_weight(class_weight='balanced', y=y_encoded)
# clf.fit(
#     X_scaled, y_encoded,
#     sample_weight=weights,                # 加权训练（可选）
#     verbose=100
# )
weights = compute_sample_weight(class_weight='balanced', y=y_encoded)

# clf = XGBClassifier(
#     objective='multi:softprob',
#     num_class=len(le.classes_),
#     max_depth=6,
#     learning_rate=0.1,
#     n_estimators=1000,
#     eval_metric='mlogloss',
#     reg_alpha=0.01,
#     reg_lambda=0.01,
#     colsample_bytree=0.6,
#     subsample=0.8,
#     random_state=42
# )

# 拟合模型，加入样本权重
# clf.fit(X_scaled, y_encoded, sample_weight=weights)

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

# 一次性批量预测（推荐）
preds = clf.predict(X_test_1)
# 转换标签回原始类别
decoded_preds = le.inverse_transform(preds.astype(int).ravel())

# 构建 predictions 字典
predictions = {keys[i]: str(decoded_preds[i]) for i in range(len(keys))}


with open('predictions1.json', "w") as z:
    z.write(str(predictions) + '\n')


# %%


# %% [markdown]
# Task2

# %%
dataroot2 = "student_files/task2_next_sequence_prediction/"

# %%
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
        # pitch_std = pitches.std()
        #new
        # pitch_median = np.median(pitches)
        pitch_min = pitches.min()
        pitch_max = pitches.max()
        pitch_mode = np.argmax(np.bincount(pitches))
        pitch_mode_count = np.bincount(pitches)[pitch_mode]
        pitch_mode_ratio = pitch_mode_count / len(pitches)
        pitch_skew = scipy.stats.skew(pitches)
        # pitch_kurtosis = scipy.stats.kurtosis(pitches)
        # pitch_hist, _ = np.histogram(pitches, bins=range(21, 109, 12))  # 按八度划分
        # note_repeats = np.sum(np.diff(pitches) == 0)
        # repeat_ratio = note_repeats / len(pitches)
        low = np.sum(pitches < 50)
        mid = np.sum((pitches >= 50) & (pitches <= 70))
        # high = np.sum(pitches > 70)

        # pitch_counts = Counter(pitches)
        # pitch_probs = np.array(list(pitch_counts.values())) / len(pitches)
        # if pitch_probs.sum() > 0:
        #     pitch_entropy = scipy.stats.entropy(pitch_probs, base=2)
        # else:
        #     pitch_entropy = 0

        # 节奏特征
        durations = np.array([note.end - note.start for note in notes])
        # duration_mean = durations.mean()
        # duration_std = durations.std()
        #new
        duration_min = durations.min()
        # duration_max = durations.max()
        duration_median = np.median(durations)
        # duration_skew = scipy.stats.skew(durations)
        # duration_kurtosis = scipy.stats.kurtosis(durations)
        # duration_change = np.abs(np.diff(durations))
        # duration_change_mean = duration_change.mean()
        # duration_change_std = duration_change.std()
        short_ratio = np.sum(durations < 0.25) / len(durations)
        # long_ratio = np.sum(durations > 1.0) / len(durations)
        unique, counts = np.unique(durations.round(3), return_counts=True)
        duration_entropy = scipy.stats.entropy(counts)

        # rhythm_entropy = scipy.stats.entropy(np.bincount((durations * 4).astype(int)))  # 用近似四分音符单位的时值编码
        # acf_vals = acf(durations, nlags=10, fft=True)


        # 音符密度
        total_time = pm.get_end_time()
        note_density = len(notes) / total_time if total_time > 0 else 0
        #new
        # step = 1.0  # 1秒为一个时间段
        onset_times = np.array([note.start for note in notes])
        # segments = np.arange(0, total_time, step)
        # densities = [np.sum((onset_times >= t) & (onset_times < t + step)) for t in segments]
        # density_std = np.std(densities)
        # density_max = np.max(densities)
        # density_min = np.min(densities)
        # density_range = density_max - density_min


        onsets_sorted = np.sort(onset_times)
        inter_onset_intervals = np.diff(onsets_sorted)
        legato_ratio = np.sum(inter_onset_intervals < 0.05) / len(inter_onset_intervals)
        
        start_times = np.array([note.start for note in notes])
        intervals = np.diff(np.sort(start_times))

        # 力度特征
        velocities = np.array([note.velocity for note in notes])
        velocity_mean = velocities.mean()
        velocity_std = velocities.std()
        #new
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
        # velocity_skew = scipy.stats.skew(velocities)
        # velocity_kurtosis = scipy.stats.kurtosis(velocities)
        # strong_ratio = np.sum(velocities > 100) / len(velocities)
        # weak_ratio = np.sum(velocities < 60) / len(velocities)


        # 多音性
        # polyphony_window = np.array([
        #     np.sum((onset_times >= t) & (onset_times < t + 0.1))
        #     for t in np.arange(0, total_time, 0.1)
        # ])
        # polyphony = polyphony_window.mean()
        # max_polyphony = polyphony_window.max()
        # min_polyphony = polyphony_window.min()
        # polyphony_std = polyphony_window.std()




        # --------------------
        # 使用 music21 分析高级特征
        # --------------------
        # score = music21.converter.parse(file_path)

        # 调性分析
        # key_obj = score.analyze('key')
        # tonic = key_obj.tonic.name
        # mode = key_obj.mode
        # key_confidence = key_obj.correlationCoefficient
        # mode_map = {'major': 1, 'minor': 0}
        # tonic_map = {
        #     'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
        #     'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
        # }
        # mode_num = mode_map.get(mode, -1)
        # tonic_num = tonic_map.get(tonic, -1)

        # 节拍（拍号）
        # try:
        #     ts = score.recurse().getElementsByClass('TimeSignature')[0]
        #     time_signature = ts.ratioString
        #     numerator, denominator = map(int, time_signature.split('/'))
        # except:
        #     time_signature = 'Unknown'
        #     numerator, denominator = 0, 0

        # # 拍点强度
        # beat_strengths = [n.beatStrength for n in score.flat.notes if hasattr(n, 'beatStrength')]
        # beat_strength_mean = np.mean(beat_strengths) if beat_strengths else 0


        # # 音程
        # notes21 = []
        # for n in score.flat.notes:
        #     if isinstance(n, music21.note.Note):
        #         notes21.append(n)
        #     elif isinstance(n, music21.chord.Chord):
        #         notes21.append(n.notes[0])  # 或 n.notes[-1], n.root(), n.pitches 等看你需要
        # intervals = [music21.interval.Interval(notes21[i], notes21[i+1]) for i in range(len(notes21)-1)]
        # interval_semitones = [abs(iv.semitones) for iv in intervals]
        # # interval_variety = len(set(interval_semitones))
        # large_intervals_ratio = np.sum(np.array(interval_semitones) > 9) / len(interval_semitones)

        # # 和弦熵
        # chords = score.chordify().flat.getElementsByClass('Chord')
        # chord_names = [c.pitchedCommonName for c in chords if c.isTriad()]
        # chord_entropy = scipy.stats.entropy(list(Counter(chord_names).values()), base=2)

        # avg_chord_len = np.mean([c.quarterLength for c in chords]) if chords else 0
        # chord_variety = len(set(chord_names))

        # 小节结构
        # try:
        #     measures = score.parts[0].getElementsByClass('Measure')
        #     note_counts_per_measure = [len(m.notes) for m in measures]
            # note_density_var = np.std(note_counts_per_measure)
            # phrase_lengths = [len(m.notes) for m in measures] if measures else []
            # phrase_std = np.std(phrase_lengths) if phrase_lengths else 0
        # except:
        #     note_density_var = 0.0

        # 休止比例
        # rests = score.flat.getElementsByClass('Rest')
        # rest_duration = sum(r.duration.quarterLength for r in rests)
        # total_duration = score.duration.quarterLength
        # rest_ratio = rest_duration / total_duration if total_duration > 0 else 0

        # offbeats = [n.beat % 1 > 0.4 for n in score.flat.notes]
        # offbeat_ratio = np.sum(offbeats) / len(offbeats) if offbeats else 0


        features = {
            'pitch_range': pitch_range,
            'pitch_mean': pitch_mean,
            # 'pitch_std': pitch_std,

            # 'pitch_median': pitch_median,
            'pitch_min': pitch_min,
            'pitch_max': pitch_max,
            # 'pitch_mode': pitch_mode,
            # 'pitch_mode_count': pitch_mode_count,
            'pitch_mode_ratio': pitch_mode_ratio,
            'pitch_skew': pitch_skew,
            # 'pitch_kurtosis': pitch_kurtosis,
            # 'repeat_ratio': repeat_ratio,
            'low_pitch_ratio': low / len(pitches),
            'mid_pitch_ratio': mid / len(pitches),
            # 'high_pitch_ratio': high / len(pitches),

            # 'pitch_entropy': pitch_entropy,

            # 'duration_mean': duration_mean,
            # 'duration_std': duration_std,

            'duration_min': duration_min,
            # 'duration_max': duration_max,
            'duration_median': duration_median,
            # 'duration_skew': duration_skew,
            # 'duration_kurtosis': duration_kurtosis,
            # 'duration_change_mean': duration_change_mean,
            # 'duration_change_std': duration_change_std,
            'short_duration_ratio': short_ratio,
            # 'long_duration_ratio': long_ratio,
            'duration_entropy': duration_entropy,
            
            # 'rhythm_entropy': rhythm_entropy,
            # 'duration_acf1': acf_vals[1],

            'note_density': note_density,
            # 'density_std' : density_std,
            # 'density_max' : density_max,
            # 'density_min' : density_min,
            # 'density_range' : density_range,

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
            # 'velocity_skew': velocity_skew,
            # 'velocity_kurtosis': velocity_kurtosis,
            # 'strong_velocity_ratio': strong_ratio,
            # 'weak_velocity_ratio': weak_ratio,

            # 'polyphony': polyphony,

            # 'max_polyphony' : max_polyphony,
            # 'min_polyphony' : min_polyphony,
            # 'polyphony_std' : polyphony_std,

            # 'mode_num': mode_num,
            # 'tonic_num': tonic_num,
            # 'key_confidence': key_confidence,

            # Music21 additional
            # 'interval_variety': interval_variety,
            # 'large_intervals_ratio': large_intervals_ratio,
            # 'chord_entropy': chord_entropy,

            # 'avg_chord_len': avg_chord_len,
            # 'chord_variety': chord_variety,
            
            # 'note_density_var': note_density_var,
            # 'phrase_std': phrase_std,
            
            # 'rest_ratio': rest_ratio,
            # 'beat_strength_mean': beat_strength_mean,
            # 'numerator' : numerator,
            # 'denominator' : denominator,

            # 'offbeat_ratio': offbeat_ratio,

        }

        # for i, val in enumerate(pitch_hist):
        #     features[f'pitch_hist_octave_{i}'] = val

        # pitch_classes = [note.pitch % 12 for note in notes]
        # pitch_classes = np.array(pitch_classes)
        # pitch_class_hist = np.bincount(pitch_classes, minlength=12)
        # for i in range(12):
        #     features[f'pc_{i}'] = pitch_class_hist[i] / np.sum(pitch_class_hist)
        # features['pitch_class_entropy'] = scipy.stats.entropy(pitch_class_hist / np.sum(pitch_class_hist))

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# %%
d = eval(open(dataroot2 + "/train.json", 'r').read())
X, y = [], []

for k in tqdm(d):
    path1, path2 = k
    x1 = extract_features(dataroot2 + path1)
    x2 = extract_features(dataroot2 + path2)

    feature_diff = [abs(a - b) for a, b in zip(x1.values(), x2.values())]

    X.append(feature_diff)
    y.append(int(d[k]))

X = np.array(X)
y = np.array(y)


# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)  

# %%
# K-Fold 交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_y_true = []
all_y_pred = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_encoded)):
    print(f"\n--- Fold {fold + 1} ---")

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    model = LGBMClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        objective='binary',
        random_state=42,
        reg_alpha=0.1,
        reg_lambda=1,
        feature_fraction = 0.8
    )

    model.fit(
        X_train, y_train
    )

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Fold {fold + 1} Accuracy: {acc:.4f}")

    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)
print("all_acc", accuracy_score(all_y_true, all_y_pred))


# %%
d = eval(open(dataroot2 + "/test.json", 'r').read())
X, y = [], []

for k in tqdm(d):
    path1, path2 = k
    x1 = extract_features(dataroot2 + path1)
    x2 = extract_features(dataroot2 + path2)

    feature_diff = [abs(a - b) for a, b in zip(x1.values(), x2.values())]

    X.append(feature_diff)

X_test = np.array(X)

# %%
d = eval(open(dataroot2 + "/test.json", 'r').read())
keys = []
for k in tqdm(d):
    try:
        keys.append(k)  # 保存文件名顺序
    except Exception as e:
        print(f"Error on {k}: {e}")

# %%
X_test_scaled = scaler.transform(X_test)

# %%
model.fit(X_scaled, y_encoded)
# 一次性批量预测（推荐）
preds = model.predict(X_test_scaled)
# 转换标签回原始类别
decoded_preds = le.inverse_transform(preds.astype(int).ravel())

# 构建 predictions 字典
predictions = {keys[i]: bool(decoded_preds[i]) for i in range(len(keys))}


with open('predictions2.json', "w") as z:
    z.write(str(predictions) + '\n')

# %% [markdown]
# Task3

# %%
TAGS = ['rock', 'oldies', 'jazz', 'pop', 'dance',  'blues',  'punk', 'chill', 'electronic', 'country']

# %%
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

# %%
# Some constants (you can change any of these if useful)
SAMPLE_RATE = 16000
N_MELS = 64
N_CLASSES = 10
AUDIO_DURATION = 10 # seconds
BATCH_SIZE = 4

# %%
dataroot3 = "student_files/task3_audio_classification/"

# %%
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

# %%

import torchaudio.transforms as T
import random
import torchaudio

def augment_waveform(waveform, sample_rate=SAMPLE_RATE):
    # 1. 随机加高斯噪声
    noise_amp = random.uniform(0.001, 0.01)
    noise = noise_amp * torch.randn_like(waveform)
    waveform = waveform + noise

    # 2. 随机变速 (Time Stretch，不变音高)
    # if random.random() < 0.3:
    #     rate = random.uniform(0.8, 1.2)
    #     new_sr = int(sample_rate * rate)
    #     waveform = torchaudio.functional.resample(waveform, sample_rate, new_sr)
    #     # 保持长度一致，裁剪或补零
    #     target_len = int(sample_rate * AUDIO_DURATION)
    #     if waveform.shape[-1] > target_len:
    #         waveform = waveform[..., :target_len]
    #     else:
    #         pad_len = target_len - waveform.shape[-1]
    #         waveform = torch.nn.functional.pad(waveform, (0, pad_len))

    # 3. 随机变调 (Pitch Shift)

    # n_steps = random.uniform(-2, 2)  # 半音上下浮动
    # try:
    #     waveform = torchaudio.functional.pitch_shift(
    #         waveform, sample_rate, n_steps
    #     )
    # except Exception:
    #     pass  # 某些 torchaudio 版本不支持 pitch_shift，可删去这段

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
        # 如果正好等长不用动

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

            # 构建扁平索引：用于__len__ 和 __getitem__
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


# %%


# %%
class Loaders():
    def __init__(self, train_path, test_path, split_ratio=0.9, seed = 0):
        torch.manual_seed(seed)
        random.seed(seed)
        
        meta_train = eval(open(train_path, 'r').read())
        l_test = eval(open(test_path, 'r').read())
        meta_test = dict([(x,[]) for x in l_test]) # Need a dictionary for the above class
        
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


# %%
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
        return self.fc2(x)  # multilabel → sigmoid

# %%
class CNNRNNClassifier(nn.Module):
    def __init__(self, n_classes, n_mels=64):
        super().__init__()

        # ✅ CNN部分：提取时频特征
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, n_mels/2, time/2]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # [B, 64, n_mels/4, time/4]
        )

        # ✅ RNN部分：建模时间依赖
        self.rnn_input_dim = (n_mels // 4) * 64
        self.rnn = nn.GRU(input_size=self.rnn_input_dim, hidden_size=128, batch_first=True, bidirectional=True)

        # ✅ FC层：分类
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        # x: [B, 1, n_mels, time]
        x = self.conv_block(x)  # [B, C=64, mel/4, time/4]
        b, c, h, t = x.shape

        # 转换为 RNN 输入格式：[B, time, features]
        x = x.permute(0, 3, 1, 2)       # [B, time, C, H]
        x = x.contiguous().view(b, t, -1)  # [B, time, C*H]

        # RNN 输出
        x, _ = self.rnn(x)  # [B, time, 256]
        x = x.mean(dim=1)   # 全时间平均池化 [B, 256]

        return self.fc(x)  # raw logits

# %%
class Pipeline():
    def __init__(self, model, learning_rate, seed = 0):
        # These two lines will (mostly) make things deterministic.
        # You're welcome to modify them to try to get a better solution.
        torch.manual_seed(seed)
        random.seed(seed)

        # self.device = torch.device("cpu") # Can change this if you have a GPU, but the autograder will use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device) #model.cuda() # Also uncomment these lines for GPU
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

    def evaluate(self, loader, threshold=0.5, outpath=None):
        self.model.eval()
        preds, targets, paths = [], [], []
        with torch.no_grad():
            for x, y, ps in loader:
                x = x.to(self.device) #x.cuda()
                y = y.to(self.device) #y.cuda()
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
            predictions[paths[i]] = [TAGS[j] for j in range(len(preds_bin[i])) if preds_bin[i][j]]
        
        mAP = None
        if outpath: # Save predictions
            with open(outpath, "w") as z:
                z.write(str(predictions) + '\n')
        else: # Only compute accuracy if we're *not* saving predictions, since we can't compute test accuracy
            mAP = average_precision_score(targets, preds, average='macro')
        return predictions, mAP

    def train(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for x, y, path in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                x = x.to(self.device) #x.cuda()
                y = y.to(self.device) #y.cuda()
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            val_predictions, mAP = self.evaluate(val_loader)
            print(f"[Epoch {epoch+1}] Loss: {running_loss/len(train_loader):.4f} | Val mAP: {mAP:.4f}")

# %%
loaders = Loaders(dataroot3 + "/train.json", dataroot3 + "/test.json")


# %%
model = CNNClassifier()
pipeline = Pipeline(model, 1e-4)

pipeline.train(loaders.loaderTrain, loaders.loaderValid, 25)
train_preds, train_mAP = pipeline.evaluate(loaders.loaderTrain, 0.5)
valid_preds, valid_mAP = pipeline.evaluate(loaders.loaderValid, 0.5)
test_preds, _ = pipeline.evaluate(loaders.loaderTest, 0.5, "predictions3.json")

# all_train = eval(open(dataroot3 + "/train.json").read())
# for k in valid_preds:
#     # We split our training set into train+valid
#     # so need to remove validation instances from the training set for evaluation
#     all_train.pop(k)
# acc3 = accuracy3(all_train, train_preds)
# print("Task 3 training mAP = " + str(acc3))



