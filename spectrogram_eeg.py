import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import spectrogram
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from scipy.stats import randint

BASE_PATH = Path(r"C:\abalaji\bichat\ORIGINAL_DATA\chunks_20")
LABEL_DIRS = {0: "0", 1: "1"}
FS = 256
NPERSEG = 512
NOVERLAP = 256
FREQ_MAX = 40
N_TIME_BINS = 20


def load_raw(path):
    raw = pd.read_csv(path, header=None)
    first_row_numeric = pd.to_numeric(raw.iloc[0], errors="coerce").notna().all()
    if not first_row_numeric:
        raw = pd.read_csv(path)
    return raw.values.astype(float)


def fit_time_bins(spec, n_bins):
    n_freq, n_time = spec.shape
    if n_time == n_bins:
        return spec
    if n_time > n_bins:
        return spec[:, :n_bins]
    padded = np.zeros((n_freq, n_bins))
    padded[:, :n_time] = spec
    return padded


def chunk_spectrogram_tensor(arr):
    channels = arr[:, :-1]
    label = arr[0, -1]
    specs = []
    for ch in range(channels.shape[1]):
        f, t, Sxx = spectrogram(channels[:, ch], fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
        keep = f <= FREQ_MAX
        spec = 10 * np.log10(Sxx[keep] + 1e-12)
        specs.append(fit_time_bins(spec, N_TIME_BINS))
    tensor = np.stack(specs, axis=0)
    return tensor, label


def build_features():
    X_flat, X_tensor, labels, patient_ids = [], [], [], []
    for label, subdir in LABEL_DIRS.items():
        folder = BASE_PATH / subdir
        for path in sorted(folder.glob("*.csv")):
            arr = load_raw(path)
            tensor, file_label = chunk_spectrogram_tensor(arr)
            X_tensor.append(tensor)
            X_flat.append(tensor.flatten())
            labels.append(label)
            patient_ids.append(path.stem.split("_chunk_")[0])
    X_flat = np.vstack(X_flat)
    X_tensor = np.stack(X_tensor)
    y = np.array(labels)
    patient_ids = np.array(patient_ids)
    np.save("spectrogram_features_flat.npy", X_flat)
    np.save("spectrogram_features_tensor.npy", X_tensor)
    np.save("spectrogram_labels.npy", y)
    np.save("spectrogram_patient_ids.npy", patient_ids)
    print("built features:", X_flat.shape, "labels:", y.shape, "patients:", len(np.unique(patient_ids)))
    return X_flat, y, patient_ids


flat_path = Path("spectrogram_features_flat.npy")
if flat_path.exists():
    X = np.load(flat_path)
    y = np.load("spectrogram_labels.npy")
    patient_ids = np.load("spectrogram_patient_ids.npy")
else:
    X, y, patient_ids = build_features()

PARAM_DIST = {
    "n_estimators": randint(100, 500),
    "max_depth": randint(3, 30),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", None],
}
N_ITER = 30
INNER_FOLDS = 5

logo = LeaveOneGroupOut()
chunk_true, chunk_pred = [], []
patient_true, patient_pred = [], []

for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=patient_ids)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    held_out_patient = patient_ids[test_idx][0]

    base_rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        base_rf,
        PARAM_DIST,
        n_iter=N_ITER,
        scoring="balanced_accuracy",
        cv=StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_rf = search.best_estimator_

    preds = best_rf.predict(X_test)
    chunk_true.extend(y_test)
    chunk_pred.extend(preds)

    majority_vote = 1 if preds.sum() > len(preds) / 2 else 0
    true_label = y_test[0]
    patient_true.append(true_label)
    patient_pred.append(majority_vote)

    print(f"fold {fold_i} patient {held_out_patient} true={int(true_label)} pred={majority_vote} chunk_acc={(preds == y_test).mean():.3f}")

chunk_true = np.array(chunk_true)
chunk_pred = np.array(chunk_pred)
patient_true = np.array(patient_true)
patient_pred = np.array(patient_pred)

print("\nchunk-level balanced accuracy:", balanced_accuracy_score(chunk_true, chunk_pred))
print("chunk-level confusion matrix:\n", confusion_matrix(chunk_true, chunk_pred))
print("\npatient-level balanced accuracy:", balanced_accuracy_score(patient_true, patient_pred))
print("patient-level confusion matrix:\n", confusion_matrix(patient_true, patient_pred))