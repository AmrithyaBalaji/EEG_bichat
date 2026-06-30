import os
import re
import logging
import numpy as np
import pandas as pd
from scipy.signal import butter, freqz, filtfilt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("loo_pipeline")

FOLDER_0      = r"D:\abalaji\chunks\0"
FOLDER_1      = r"D:\abalaji\chunks\1"
OUTPUT_DIR    = r"D:\abalaji\PROCESSED_LOO_60"

FILTER_ORDER  = 4
FILTER_CUTOFF = 100.0
FILTER_FS     = 256.0
FFT_FS        = 256
INTERP_WINDOW = 64
LABEL_COL     = "label"
EXCEPTION_COL = "label"


def butter_lowpass_filter(signal, order, cutoff, fs):
    signal = pd.to_numeric(signal, errors='coerce').to_numpy(dtype=np.float64)
    if np.isnan(signal).any():
        raise ValueError(f"Signal contains NaN after numeric coercion — {np.isnan(signal).sum()} non-numeric value(s) found.")
    fs = int(fs)
    nyq = 0.5 * fs
    midcut = cutoff / nyq
    b, a = butter(order, midcut, btype="lowpass")
    w, h = freqz(b, a)
    return filtfilt(b, a, signal)


def fast_fourier(signal, freq):
    N = len(signal)
    window = np.hanning(N)
    signal_windowed = signal * window
    fft_vals = np.fft.fft(signal_windowed)
    fft_magnitude = np.abs(fft_vals) / N
    freqs = np.fft.fftfreq(N, d=1/freq)
    return fft_magnitude[:N//2], freqs[:N//2]


def merge_all_columns_to_mean(df, except_column=""):
    excepted_column = pd.DataFrame()
    if except_column:
        for col in df.columns:
            if except_column in col:
                except_column = col
        excepted_column = df[except_column]
        df.drop(except_column, axis=1, inplace=True)

    df_mean = pd.DataFrame(columns=["mean"])
    df_mean['mean'] = df.mean(axis=1)

    if except_column != "":
        for col in df.columns:
            if except_column in col:
                except_column = col
        df_mean[except_column] = excepted_column

    return df_mean


def smoothing(data, n, mode='mean'):
    if len(data) > n:
        old_size = len(data)
        x_old = np.arange(old_size)
        x_new = np.linspace(0, old_size - 1, n)
        return np.interp(x_new, x_old, data)
    else:
        raise Exception("smoothing: length of data " + str(len(data)) + " < n " + str(n))


def step_filter(df):
    sig_cols = [c for c in df.columns if EXCEPTION_COL not in c] if EXCEPTION_COL else list(df.columns)
    for ch in sig_cols:
        df.loc[:, ch] = butter_lowpass_filter(df[ch], FILTER_ORDER, FILTER_CUTOFF, FILTER_FS)
    return df


def step_fft(df):
    sig_cols = [c for c in df.columns if EXCEPTION_COL not in c] if EXCEPTION_COL else list(df.columns)
    fft_df = pd.DataFrame()
    for ch in sig_cols:
        clean_fft, clean_freqs = fast_fourier(df[ch], FFT_FS)
        if "Frequency [Hz]" not in fft_df.columns:
            fft_df["Frequency [Hz]"] = clean_freqs
        fft_df[ch] = clean_fft
    return fft_df


def step_average(df):
    return merge_all_columns_to_mean(df, "Frequency [Hz]").round(3)


def step_linear_interpolation(df):
    df_out = pd.DataFrame()
    for ch in df.columns:
        df_out[ch] = smoothing(df[ch], INTERP_WINDOW, 'mean')
    return df_out


def process_chunk(df):
    df = step_filter(df)
    df = step_fft(df)
    df = step_average(df)
    df = step_linear_interpolation(df)
    return df


def discover_files(folder0, folder1):
    pattern = re.compile(r"^(\d+)_chunk_\d+\.csv$", re.IGNORECASE)
    files_by_id = {}
    for label, folder in [(0, folder0), (1, folder1)]:
        if not os.path.isdir(folder):
            logger.warning(f"Folder not found: {folder}")
            continue
        for fname in os.listdir(folder):
            m = pattern.match(fname)
            if m:
                sid = m.group(1)
                files_by_id.setdefault(sid, []).append((os.path.join(folder, fname), label))
    return files_by_id


def process_group(file_list):
    rows = []
    for fpath, label in file_list:
        try:
            df_proc = process_chunk(pd.read_csv(fpath, index_col=False).astype(float))
            mean_row = df_proc["mean"].to_numpy()
            row_dict = {f"feature_{i}": mean_row[i] for i in range(len(mean_row))}
            row_dict[LABEL_COL] = label
            rows.append(row_dict)
        except Exception as e:
            logger.error(f"  Error processing {fpath}: {e}")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def main():
    train_dir = os.path.join(OUTPUT_DIR, "0")
    test_dir  = os.path.join(OUTPUT_DIR, "1")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    logger.info("Discovering files …")
    files_by_id = discover_files(FOLDER_0, FOLDER_1)

    if not files_by_id:
        logger.error("No files found. Check FOLDER_0 and FOLDER_1 paths.")
        return

    sample_ids = sorted(files_by_id.keys())
    logger.info(f"Found {sum(len(v) for v in files_by_id.values())} total chunks across {len(sample_ids)} sample IDs: {sample_ids}")

    all_train_parts, all_test_parts = [], []

    for idx, test_id in enumerate(sample_ids, 1):
        logger.info(f"\n[{idx}/{len(sample_ids)}] LOO split — test sample: {test_id}")

        test_files  = files_by_id.get(test_id, [])
        train_files = [(fp, lb) for sid, fl in files_by_id.items() if sid != test_id for fp, lb in fl]

        logger.info(f"  Test  ID={test_id}: {len(test_files)} chunk(s)")
        logger.info(f"  Train IDs≠{test_id}: {len(train_files)} chunk(s)")

        train_df = process_group(train_files)
        test_df  = process_group(test_files)

        if not train_df.empty:
            train_path = os.path.join(train_dir, f"train_without_{test_id}.csv")
            train_df.to_csv(train_path, index=False)
            logger.info(f"  Saved → {train_path}  ({len(train_df)} rows, {train_df.shape[1]} cols)")
            all_train_parts.append(train_df)

        if not test_df.empty:
            test_path = os.path.join(test_dir, f"test_{test_id}.csv")
            test_df.to_csv(test_path, index=False)
            logger.info(f"  Saved → {test_path}  ({len(test_df)} rows, {test_df.shape[1]} cols)")
            all_test_parts.append(test_df)

    if all_train_parts:
        full_train = pd.concat(all_train_parts, ignore_index=True)
        p = os.path.join(train_dir, "full_train_dataset.csv")
        full_train.to_csv(p, index=False)
        logger.info(f"\nFull train dataset saved → {p}  ({len(full_train)} rows)")

    if all_test_parts:
        full_test = pd.concat(all_test_parts, ignore_index=True)
        p = os.path.join(test_dir, "full_test_dataset.csv")
        full_test.to_csv(p, index=False)
        logger.info(f"Full test  dataset saved → {p}  ({len(full_test)} rows)")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()