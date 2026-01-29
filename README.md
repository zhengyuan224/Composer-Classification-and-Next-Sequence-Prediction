# Composer Classification and Next Sequence Prediction

This project focuses on music analysis using machine learning and deep learning techniques. It covers three primary tasks: classifying composers from MIDI files, predicting the sequence of music segments, and multi-label tagging of audio genres.

## 🎵 Project Overview

The project leverages MIDI and raw audio features to perform sophisticated music information retrieval (MIR) tasks. By extracting pitch, rhythm, and timbre information, we can build models that understand the stylistic signatures of different composers and genres.

## 🚀 Tasks

### 1. Composer Classification
**Goal**: Identify the composer of a given MIDI file.
- **Features**: Pitch range, mean/median frequency, note density, velocity variations, legato ratio, chord entropy, and time signature analysis.
- **Models**: Uses high-performance gradient boosting frameworks including **LightGBM**, **XGBoost**, and **CatBoost**.
- **Techniques**: feature selection based on importance (gain), stratified K-fold cross-validation, and label encoding.

### 2. Next Sequence Prediction
**Goal**: Determine if two music segments follow each other in a sequence.
- **Approach**: Feature difference analysis between consecutive segments.
- **Features**: Analyzes the delta in pitch range, duration entropy, and velocity changes between paths.
- **Model**: Binary classification using **LightGBM**.

### 3. Audio Genre Tagging
**Goal**: Multi-label classification to assign genre tags (e.g., rock, jazz, pop, blues) to audio files.
- **Preprocessing**: Audio files are transformed into **MelSpectrograms** and converted to Decibel units.
- **Model Architecture**: 
  - **CNN**: 2D Convolutional layers for spatial feature extraction from spectrograms.
  - **CRNN**: (Hybrid) Combining CNN for feature extraction and GRU for temporal modeling.
- **Training**: Data augmentation (noise injection, random silence, spec-augment) and BCEWithLogitsLoss for multi-label prediction.

## 🛠️ Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install torch torchaudio librosa pretty_midi music21 pandas numpy scikit-learn lightgbm xgboost catboost tqdm miditoolkit soundfile
```

## 📂 Project Structure

- `task1_composer.py`: Implementation for composer classification.
- `task2_sequence.py`: Implementation for sequence prediction.
- `task3_audio.py`: Implementation for audio genre tagging.
- `all_code.py`: Unified script containing all experiments and model pipelines.

## 📊 Results & Evaluation

- **Task 1 & 2**: Evaluated using Accuracy and F1-score with K-fold cross-validation.
- **Task 3**: Evaluated using mean Average Precision (mAP) for multi-label tagging.

## 📝 Usage

To run a specific task, execute the corresponding script:

```bash
python task1_composer.py
python task2_sequence.py
python task3_audio.py
```

Predictions are saved as JSON files (e.g., `predictions1.json`, `predictions2.json`, `predictions3.json`).
