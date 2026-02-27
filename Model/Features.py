import os
import librosa
import numpy as np
import pandas as pd

# ===== CONFIG =====
dataset_split_path = r"E:\InfantCryClassifier\DatasetSplit"  # train/val/test folders
output_dir = r"E:\InfantCryClassifier\Features"
classes = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
splits = ['train', 'val', 'test']
sr = 22050
fixed_duration = 3  # seconds
os.makedirs(output_dir, exist_ok=True)

# ===== FEATURE EXTRACTION FUNCTION =====
def extract_features(file_path, sr=22050):
    y, _ = librosa.load(file_path, sr=sr)
    
    # --- Trim/Pad to fixed duration ---
    desired_len = int(fixed_duration * sr)
    if len(y) > desired_len:
        y = y[:desired_len]
    else:
        y = np.pad(y, (0, max(0, desired_len - len(y))))
    
    # --- Time-Frequency Features ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_mean = np.mean(librosa.power_to_db(mel_spec, ref=np.max), axis=1)
    
    stft = np.abs(librosa.stft(y))
    spec_mean = np.mean(stft, axis=1)[:40]
    
    # --- Deep Learning Features ---
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    
    return np.hstack([mfccs_mean, mel_mean, spec_mean, zcr, centroid, rms])

# ===== PROCESS SPLITS =====
for split in splits:
    features = []
    labels = []
    
    for label in classes:
        label_path = os.path.join(dataset_split_path, split, label)
        if not os.path.isdir(label_path):
            print(f"⚠️ Warning: {label_path} does not exist")
            continue
        
        for file in os.listdir(label_path):
            if not file.lower().endswith(".wav"):
                continue
            file_path = os.path.join(label_path, file)
            try:
                feat = extract_features(file_path, sr)
                features.append(feat)
                labels.append(label)
            except Exception as e:
                print(f"⚠️ Skipped {file_path}: {e}")
    
    # Save CSV per split
    df = pd.DataFrame(features)
    df['label'] = labels
    output_csv = os.path.join(output_dir, f"{split}.csv")
    df.to_csv(output_csv, index=False)
    print(f"✅ {split} features saved to: {output_csv}")

print("✅ All splits and feature extraction completed!")