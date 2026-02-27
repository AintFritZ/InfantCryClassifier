import os
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
import random

# ===== CONFIGURATION =====
dataset_path = r"E:\Dataset\donateacry_corpus"  # Original dataset
output_path = r"E:\InfantCryClassifier\ProcessedDataset"           # Processed dataset
fixed_duration = 5  # seconds
augmentations_per_file = 2  # how many augmented versions per file

# Create output folder if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# ===== UTILITY FUNCTIONS =====
def normalize_audio(y):
    return y / max(abs(y)) if max(abs(y)) > 0 else y

def trim_or_pad(y, sr, duration):
    desired_length = int(sr * duration)
    if len(y) > desired_length:
        return y[:desired_length]
    else:
        return np.pad(y, (0, max(0, desired_length - len(y))))

def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    augmented = y + noise_factor * noise
    return normalize_audio(augmented)

def pitch_shift(y, sr, n_steps=None):
    if n_steps is None:
        n_steps = random.choice([-2, -1, 1, 2])
    return librosa.effects.pitch_shift(y, sr, n_steps=n_steps)

def time_stretch(y, rate=None):
    if rate is None:
        rate = random.uniform(0.9, 1.1)
    return librosa.effects.time_stretch(y, rate)

# ===== PROCESSING =====
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    if not os.path.isdir(label_path):
        continue

    output_label_path = os.path.join(output_path, label)
    os.makedirs(output_label_path, exist_ok=True)

    for file in os.listdir(label_path):
        if not file.lower().endswith(".wav"):
            continue

        file_path = os.path.join(label_path, file)
        y, sr = librosa.load(file_path, sr=None)

        # 1. Normalize
        y = normalize_audio(y)

        # 2. Noise Reduction
        y = nr.reduce_noise(y=y, sr=sr)

        # 3. Trim / Pad
        y = trim_or_pad(y, sr, fixed_duration)

        # Save original processed file
        out_file = os.path.join(output_label_path, file)
        sf.write(out_file, y, sr)

        # 4. Data Augmentation
        for i in range(augmentations_per_file):
            y_aug = y.copy()
            choice = random.choice(["pitch", "stretch", "noise"])
            if choice == "pitch":
                y_aug = pitch_shift(y_aug, sr)
            elif choice == "stretch":
                y_aug = time_stretch(y_aug)
                y_aug = trim_or_pad(y_aug, sr, fixed_duration)  # adjust length after stretching
            elif choice == "noise":
                y_aug = add_noise(y_aug)

            aug_file_name = file.replace(".wav", f"_aug{i+1}.wav")
            sf.write(os.path.join(output_label_path, aug_file_name), y_aug, sr)

print("✅ Preprocessing and augmentation completed!")