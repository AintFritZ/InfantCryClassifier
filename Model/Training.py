import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras.layers import Reshape, Bidirectional, LSTM, Dense, Flatten
from tensorflow.keras.utils import to_categorical

# ===== CONFIG =====
dataset_split_path = r"E:\InfantCryClassifier\DatasetSplit"
classes = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
splits = ['train', 'val', 'test']
sr = 22050
segment_duration = 5  # seconds
n_mels = 40
batch_size = 32
epochs = 50

# ===== FUNCTION TO EXTRACT MEL-SPECTROGRAM FEATURE =====
def extract_mel_spectrogram(file_path, sr=22050, duration=3, n_mels=40):
    y, _ = librosa.load(file_path, sr=sr)
    desired_len = sr * duration
    if len(y) > desired_len:
        y = y[:desired_len]
    else:
        y = np.pad(y, (0, max(0, desired_len - len(y))))
    
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_db.T  # transpose to (time_steps, n_mels)

# ===== LOAD DATA =====
def load_dataset(split):
    X, y = [], []
    for label in classes:
        label_path = os.path.join(dataset_split_path, split, label)
        if not os.path.isdir(label_path):
            continue
        for file in os.listdir(label_path):
            if not file.lower().endswith(".wav"):
                continue
            file_path = os.path.join(label_path, file)
            mel = extract_mel_spectrogram(file_path, sr, segment_duration, n_mels)
            X.append(mel)
            y.append(label)
    X = np.array(X)[..., np.newaxis]  # add channel dimension for Conv2D
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y, num_classes=len(classes))
    return X, y

X_train, y_train = load_dataset('train')
X_val, y_val = load_dataset('val')
X_test, y_test = load_dataset('test')

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# ===== BUILD MODEL =====
input_shape = X_train.shape[1:]  # (time_steps, n_mels, 1)
inputs = Input(shape=input_shape)

# --- Multi-Convolutional Blocks ---
x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.3)(x)

x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.3)(x)

x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.3)(x)

# --- Reshape for BiLSTM ---
shape = x.shape
x = Reshape((shape[1], shape[2]*shape[3]))(x)

# --- BiLSTM Layer ---
x = Bidirectional(LSTM(128, return_sequences=False))(x)
x = Dropout(0.3)(x)

# --- Dense Layers ---
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)

# --- Multi-class output (Cry triggers) ---
outputs = Dense(len(classes), activation='softmax', name='cry_trigger')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ===== TRAIN MODEL =====
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs
)

# ===== EVALUATE =====
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

# ===== SAVE MODEL =====
save_dir = r"E:\InfantCryClassifier\saves"
os.makedirs(save_dir, exist_ok=True)  # create folder if it doesn't exist

# Save in native Keras format (recommended)
model_save_path = os.path.join(save_dir, "cnn_bilstm_multitask.keras")
model.save(model_save_path)
print(f"✅ Model saved to: {model_save_path}")