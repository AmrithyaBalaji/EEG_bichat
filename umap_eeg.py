import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt

BASE_PATH = Path(r"C:\abalaji\bichat\ORIGINAL_DATA\chunks_20")
LABEL_DIRS = {0: "0", 1: "1"}
FILTER_ORDER = 4
FILTER_CUTOFF = 40
FILTER_FS = 256
N_INTERP_POINTS = 64


def butter_lowpass_filter(signal, order, cutoff, fs):
    signal = pd.to_numeric(pd.Series(signal), errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(signal).any():
        raise ValueError(f"Signal contains NaN after numeric coercion — {np.isnan(signal).sum()} non-numeric value(s) found.")
    fs = int(fs)
    nyq = 0.5 * fs
    midcut = cutoff / nyq
    b, a = butter(order, midcut, btype="lowpass")
    return filtfilt(b, a, signal)


def fast_fourier(signal, freq):
    N = len(signal)
    window = np.hanning(N)
    signal_windowed = signal * window
    fft_vals = np.fft.fft(signal_windowed)
    fft_magnitude = np.abs(fft_vals) / N
    freqs = np.fft.fftfreq(N, d=1 / freq)
    return fft_magnitude[:N // 2], freqs[:N // 2]


def load_raw(path):
    raw = pd.read_csv(path, header=None)
    first_row_numeric = pd.to_numeric(raw.iloc[0], errors="coerce").notna().all()
    if not first_row_numeric:
        raw = pd.read_csv(path)
    return raw.values.astype(float)


def chunk_to_feature(arr):
    channels = arr[:, :-1]
    file_label = arr[0, -1]
    spectra = []
    for ch in range(channels.shape[1]):
        filtered = butter_lowpass_filter(channels[:, ch], FILTER_ORDER, FILTER_CUTOFF, FILTER_FS)
        magnitude, _ = fast_fourier(filtered, FILTER_FS)
        spectra.append(magnitude)
    avg_spectrum = np.mean(spectra, axis=0)
    x_old = np.linspace(0, 1, len(avg_spectrum))
    x_new = np.linspace(0, 1, N_INTERP_POINTS)
    interpolated = np.interp(x_new, x_old, avg_spectrum)
    return interpolated, file_label


rows = []
labels = []
patient_ids = []
label_mismatches = 0

for label, subdir in LABEL_DIRS.items():
    folder = BASE_PATH / subdir
    for f in sorted(folder.glob("*.csv")):
        arr = load_raw(f)
        feature_vec, file_label = chunk_to_feature(arr)
        if file_label != label:
            label_mismatches += 1
        rows.append(feature_vec)
        labels.append(label)
        patient_ids.append(f.stem.split("_chunk_")[0])

print("label mismatches vs folder:", label_mismatches)

X = np.vstack(rows)
y = np.array(labels)
patient_ids = np.array(patient_ids)

print("total chunk-level points:", X.shape[0])
print("total unique patients:", len(np.unique(patient_ids)))

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean", random_state=42)
embedding = reducer.fit_transform(X)

fig, ax = plt.subplots(figsize=(8, 6))
for label in np.unique(y):
    mask = y == label
    ax.scatter(embedding[mask, 0], embedding[mask, 1], s=12, alpha=0.7, label=str(label))
ax.legend(title="label")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title("UMAP of EEG chunks (filtered + FFT + channel-avg + 64pt interp)")
plt.tight_layout()
plt.savefig("umap_eeg_fft_chunks.png", dpi=200)
plt.show()

df_patient = pd.DataFrame(embedding, columns=["umap_1", "umap_2"])
df_patient["patient_id"] = patient_ids
df_patient["label"] = y
patient_means = df_patient.groupby("patient_id").agg(
    umap_1=("umap_1", "mean"),
    umap_2=("umap_2", "mean"),
    label=("label", "first"),
).reset_index()

fig2, ax2 = plt.subplots(figsize=(8, 6))
for label in np.unique(patient_means["label"]):
    mask = patient_means["label"] == label
    ax2.scatter(patient_means.loc[mask, "umap_1"], patient_means.loc[mask, "umap_2"], s=40, alpha=0.8, label=str(label))
ax2.legend(title="label")
ax2.set_xlabel("UMAP 1")
ax2.set_ylabel("UMAP 2")
ax2.set_title("UMAP, chunk-mean per patient")
plt.tight_layout()
plt.savefig("umap_eeg_fft_patients.png", dpi=200)
plt.show()