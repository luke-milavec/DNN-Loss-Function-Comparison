import os
import numpy as np
from glob import glob
from collections import Counter
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Configurable paths
DATA_DIR = "data\G12EC\WFDB"
OUTPUT_DIR = "data\G12EC\processed"
DEFAULT_SIGNAL_LENGTH = 5000
SAMPLING_FREQUENCY = 500
MIN_LABEL_RATIO = 0.01
SPLIT_SEED = 42

def load_signal(mat_path):
    mat = loadmat(mat_path)
    signal = mat['val'].T  # shape: (5000, 12)
    return signal


import os
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from glob import glob

def parse_header(hea_path):
    with open(hea_path, 'r') as f:
        lines = f.readlines()

    # Extract age, sex, and diagnoses
    age, sex, dx = None, None, None
    for line in lines:
        if line.startswith('#Age'):
            age = line.strip().split(': ')[-1]
        elif line.startswith('#Sex'):
            sex = line.strip().split(': ')[-1]
        elif line.startswith('#Dx'):
            dx = line.strip().split(': ')[-1].split(',')

    return age, sex, dx

def load_signal(mat_path):
    mat = loadmat(mat_path)
    signal = mat['val'].T  # shape: (5000, 12)
    return signal

def load_data(data_dir):
    hea_files = sorted(glob(os.path.join(data_dir, "*.hea")))
    
    signals, labels, ages, sexes = [], [], [], []

    for hea_path in tqdm(hea_files):
        base = os.path.splitext(hea_path)[0]
        mat_path = base + '.mat'

        if not os.path.exists(mat_path):
            print(f"Missing .mat file for {hea_path}")
            continue

        try:
            signal = load_signal(mat_path)
            if signal.shape[1] != 12:
                continue  # skip non-12-lead
            age, sex, dx = parse_header(hea_path)
            signals.append(np.nan_to_num(signal))
            labels.append(dx)
            ages.append(int(age) if age and age.isdigit() else np.nan)
            sexes.append(1 if sex and sex.lower() == "male" else 0)
        except Exception as e:
            print(f"Error in {hea_path}: {e}")
            continue

    return signals, labels, np.array(ages), np.array(sexes)


def filter_labels(labels, threshold):
    label_counts = Counter(l for label in labels for l in label)
    valid_labels = {l for l, c in label_counts.items() if c >= threshold}
    return sorted(valid_labels)

def binarize_labels(labels, valid_labels):
    label_to_idx = {l: i for i, l in enumerate(valid_labels)}
    Y = np.zeros((len(labels), len(valid_labels)))
    for i, label_set in enumerate(labels):
        for l in label_set:
            if l in label_to_idx:
                Y[i, label_to_idx[l]] = 1
    return Y, valid_labels

def pad_signals(signals, length=DEFAULT_SIGNAL_LENGTH):
    padded = []
    for s in signals:
        if s.shape[0] < length:
            pad = np.zeros((length - s.shape[0], s.shape[1]))
            s = np.concatenate([s, pad], axis=0)
        elif s.shape[0] > length:
            s = s[:length, :]
        padded.append(s)
    return np.array(padded)

def normalize_signals(X_train, X_val, X_test):
    mean = np.mean(X_train, axis=(0, 1))
    std = np.std(X_train, axis=(0, 1))
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std

def split_data(X, Y, seed=SPLIT_SEED):
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, temp_idx = next(splitter.split(X, Y))

    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(splitter.split(X[temp_idx], Y[temp_idx]))

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[temp_idx][val_idx], Y[temp_idx][val_idx]
    X_test, Y_test = X[temp_idx][test_idx], Y[temp_idx][test_idx]

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)

def save_data(X, Y, prefix):
    np.save(os.path.join(OUTPUT_DIR, f"X_{prefix}.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, f"Y_{prefix}.npy"), Y)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    signals, labels, ages, sexes = load_data(DATA_DIR)
    X = pad_signals(signals)
    valid_labels = filter_labels(labels, MIN_LABEL_RATIO * len(signals))
    Y, label_list = binarize_labels(labels, valid_labels)
    

    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_data(X, Y)
    X_train, X_val, X_test = normalize_signals(X_train, X_val, X_test)

    save_data(X_train, Y_train, "train")
    save_data(X_val, Y_val, "val")
    save_data(X_test, Y_test, "test")
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), label_list)

    print(f"Train samples: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Number of valid labels: {len(label_list)}")

    # Simple visualization: show mean signal for lead I
    plt.plot(np.mean(X_train[:, :, 0], axis=0))
    plt.title("Mean Signal of Lead I (Train Set)")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude (normalized)")
    plt.show()

if __name__ == "__main__":
    main()
