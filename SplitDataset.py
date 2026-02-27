import os
import shutil
from sklearn.model_selection import train_test_split

# ===== CONFIG =====
dataset_path = r"E:\InfantCryClassifier\ProcessedDataset"  # your processed dataset
output_path = r"E:\InfantCryClassifier\DatasetSplit"       # where splits will be saved
classes = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
random_seed = 42

os.makedirs(output_path, exist_ok=True)

# ===== CREATE SPLIT FOLDERS =====
for split in ['train', 'val', 'test']:
    split_path = os.path.join(output_path, split)
    os.makedirs(split_path, exist_ok=True)
    for label in classes:
        os.makedirs(os.path.join(split_path, label), exist_ok=True)

# ===== SPLIT DATA =====
for label in classes:
    label_path = os.path.join(dataset_path, label)
    if not os.path.isdir(label_path):
        print(f"⚠️ Warning: {label_path} does not exist")
        continue
    
    files = [f for f in os.listdir(label_path) if f.lower().endswith(".wav")]
    
    # Split train / temp (val+test)
    train_files, temp_files = train_test_split(
        files, test_size=(1 - train_ratio), random_state=random_seed
    )
    
    # Split temp into val / test
    val_files, test_files = train_test_split(
        temp_files, test_size=test_ratio/(val_ratio + test_ratio), random_state=random_seed
    )
    
    # Copy files to respective folders
    for split_name, split_files in zip(['train','val','test'], [train_files, val_files, test_files]):
        for f in split_files:
            src = os.path.join(label_path, f)
            dst = os.path.join(output_path, split_name, label, f)
            shutil.copy2(src, dst)

print("✅ Dataset successfully split into train/val/test folders for all 5 classes!")