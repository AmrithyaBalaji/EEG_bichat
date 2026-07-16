import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import spectrogram
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from scipy.stats import randint

BASE_PATH = Path(r"C:\abalaji\bichat\ORIGINAL_DATA\chunks_20")
LABEL_DIRS = {0: "0", 1: "1"}
FS = 256
NPERSEG = 512
NOVERLAP = 256
FREQ_MAX = 40
N_TIME_BINS = 20
RANDOM_STATE = 42


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
    print("building features from raw CSVs...")
    X_flat, X_tensor, labels, patient_ids = [], [], [], []
    for label, subdir in LABEL_DIRS.items():
        folder = BASE_PATH / subdir
        files = sorted(folder.glob("*.csv"))
        print(f"label {label}: {len(files)} files in {folder}")
        for path in files:
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
    print("loading cached features from disk...")
    X = np.load(flat_path)
    y = np.load("spectrogram_labels.npy")
    patient_ids = np.load("spectrogram_patient_ids.npy")
    print("loaded:", X.shape, "labels:", y.shape, "patients:", len(np.unique(patient_ids)))
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

# ---- patient-level label (each patient's chunks share one label) ----
unique_patients = np.unique(patient_ids)
patient_label = {}
for pid in unique_patients:
    labels_for_patient = y[patient_ids == pid]
    assert len(np.unique(labels_for_patient)) == 1, f"patient {pid} has mixed labels"
    patient_label[pid] = labels_for_patient[0]

alive_patients = [pid for pid in unique_patients if patient_label[pid] == 0]
dead_patients = [pid for pid in unique_patients if patient_label[pid] == 1]
print(f"total patients: {len(unique_patients)} (alive={len(alive_patients)}, dead={len(dead_patients)})")

# --- choose test patients ---
# Option A: hardcode explicit IDs for reproducibility/control
TEST_PATIENTS = None  # e.g. ["patient_003", "patient_017", "patient_022", "patient_009"]

if TEST_PATIENTS is None:
    rng = np.random.RandomState(RANDOM_STATE)
    test_alive = rng.choice(alive_patients, size=3, replace=False).tolist()
    test_dead = rng.choice(dead_patients, size=1, replace=False).tolist()
    TEST_PATIENTS = test_alive + test_dead

print("test patients:", TEST_PATIENTS)
for pid in TEST_PATIENTS:
    print(f"  {pid}: label={patient_label[pid]}, n_chunks={(patient_ids == pid).sum()}")

test_mask = np.isin(patient_ids, TEST_PATIENTS)
train_mask = ~test_mask

X_train, y_train, train_ids = X[train_mask], y[train_mask], patient_ids[train_mask]
X_test, y_test, test_ids = X[test_mask], y[test_mask], patient_ids[test_mask]
print(f"train: {X_train.shape}, patients={len(np.unique(train_ids))}")
print(f"test:  {X_test.shape}, patients={len(np.unique(test_ids))}")

base_rf = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
search = RandomizedSearchCV(
    base_rf,
    PARAM_DIST,
    n_iter=N_ITER,
    scoring="balanced_accuracy",
    cv=StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE),
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
print("running hyperparameter search...")
search.fit(X_train, y_train)
best_rf = search.best_estimator_
print("best params:", search.best_params_)
print("best inner CV balanced accuracy:", search.best_score_)

chunk_pred = best_rf.predict(X_test)
chunk_true = y_test

patient_true, patient_pred = [], []
for pid in TEST_PATIENTS:
    pid_mask = test_ids == pid
    true_label = y_test[pid_mask][0]
    preds = chunk_pred[pid_mask]
    majority_vote = 1 if preds.sum() > len(preds) / 2 else 0
    patient_true.append(true_label)
    patient_pred.append(majority_vote)
    print(f"patient {pid} true={int(true_label)} pred={majority_vote} chunk_acc={(preds == y_test[pid_mask]).mean():.3f}")

patient_true = np.array(patient_true)
patient_pred = np.array(patient_pred)

print("\nchunk-level balanced accuracy:", balanced_accuracy_score(chunk_true, chunk_pred))
print("chunk-level confusion matrix:\n", confusion_matrix(chunk_true, chunk_pred))
print("\npatient-level balanced accuracy:", balanced_accuracy_score(patient_true, patient_pred))
print("patient-level confusion matrix:\n", confusion_matrix(patient_true, patient_pred))