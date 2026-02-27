import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras.layers import Reshape, Bidirectional, LSTM, Dense
from tensorflow.keras.utils import to_categorical

# ===== GPU CONFIGURATION =====
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ TensorFlow will use GPU: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"⚠️ Error setting memory growth: {e}")
else:
    print("⚠️ No GPU found, TensorFlow will use CPU")

# ===== CONFIG =====
dataset_split_path = r"E:\InfantCryClassifier\DatasetSplit"
classes = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
sr = 22050
segment_duration = 5  # seconds
n_mels = 40
batch_size = 32
epochs = 50
save_dir = r"E:\InfantCryClassifier\saves"
os.makedirs(save_dir, exist_ok=True)

# ===== FUNCTION TO EXTRACT MEL-SPECTROGRAM FEATURE =====
def extract_mel_spectrogram(file_path, sr=22050, duration=5, n_mels=40):
    y, _ = librosa.load(file_path, sr=sr)
    desired_len = sr * duration
    if len(y) > desired_len:
        y = y[:desired_len]
    else:
        y = np.pad(y, (0, max(0, desired_len - len(y))))
    
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_db.T

# ===== LOAD DATA WITH PROGRESS UPDATES =====
def load_dataset(split):
    X, y = [], []
    print(f"\n📥 Loading {split} data...")
    for label in classes:
        label_path = os.path.join(dataset_split_path, split, label)
        if not os.path.isdir(label_path):
            print(f"⚠️ Warning: {label_path} does not exist")
            continue
        files = [f for f in os.listdir(label_path) if f.lower().endswith('.wav')]
        print(f"➡️ Found {len(files)} files for class '{label}'")
        for i, file in enumerate(files):
            file_path = os.path.join(label_path, file)
            mel = extract_mel_spectrogram(file_path, sr, segment_duration, n_mels)
            X.append(mel)
            y.append(label)
            if (i+1) % 10 == 0 or (i+1)==len(files):
                print(f"   Processed {i+1}/{len(files)} files for class '{label}'")
    X = np.array(X)[..., np.newaxis]  # add channel dimension
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y, num_classes=len(classes))
    print(f"✅ Finished loading {split} split: {len(X)} samples")
    return X, y

# ===== LOAD DATA =====
X_train, y_train = load_dataset('train')
X_val, y_val = load_dataset('val')
X_test, y_test = load_dataset('test')

print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# ===== BUILD MODEL =====
input_shape = X_train.shape[1:]
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

# --- Multi-class output ---
outputs = Dense(len(classes), activation='softmax', name='cry_trigger')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ===== TRAIN MODEL =====
print("\n🏋️ Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs
)

# ===== EVALUATE MODEL =====
print("\n📊 Evaluating on test data...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

# ===== SAVE MODEL WITH NUMBERED FILENAME =====
existing_files = [f for f in os.listdir(save_dir) if f.startswith("cnn_bilstm_multitask")]
model_number = len(existing_files) + 1
model_save_path = os.path.join(save_dir, f"cnn_bilstm_multitask_{model_number}.keras")
model.save(model_save_path)
print(f"💾 Model saved to: {model_save_path}")

# ===== PLOT TRAINING HISTORY =====
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plot_save_path = os.path.join(save_dir, f"training_plot_{model_number}.png")
plt.savefig(plot_save_path)
plt.show()
print(f"📈 Training plot saved to: {plot_save_path}")