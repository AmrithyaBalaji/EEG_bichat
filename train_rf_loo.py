import os
import re
import logging
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("rf_loo")

TRAIN_DIR  = r"D:\abalaji\PROCESSED_LOO\0"
TEST_DIR   = r"D:\abalaji\PROCESSED_LOO\1"
OUTPUT_DIR = r"D:\abalaji\PROCESSED_LOO\results"
LABEL_COL    = "label"
HPO_N_FOLDS  = 50
HPO_CV       = 5
RANDOM_SEED  = 42


USE_SMOTE           = True
REMOVE_OUTLIERS     = True
RUN_BOTH_CONDITIONS = True


OUTLIER_IQR_K        = 1.5

HPO_PARAM_GRID = {
    "n_estimators":      [100, 200, 300, 500],
    "max_depth":         [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2"],
}


def select_hpo_folds(folds, n_folds, seed=RANDOM_SEED):
    """Randomly sample n_folds folds (without replacement) to use for HPO,
    instead of always taking the first n_folds in directory order."""
    rng = np.random.RandomState(seed)
    n_folds = min(n_folds, len(folds))
    idx = rng.choice(len(folds), size=n_folds, replace=False)
    idx.sort()
    return [folds[i] for i in idx]


def remove_outliers_iqr(X, y, k=OUTLIER_IQR_K):
    """
    Remove rows from X (and matching y) that are outliers in ANY feature,
    using the IQR rule: outlier if value < Q1 - k*IQR or > Q3 + k*IQR.
    """
    X = np.asarray(X, dtype=float)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr


    mask = np.all((X >= lower) & (X <= upper), axis=1)
    n_removed = (~mask).sum()
    if n_removed > 0:
        logger.info(f"    Outlier removal: dropped {n_removed}/{len(X)} rows")
    return X[mask], np.asarray(y)[mask]


def apply_smote(X, y, seed=RANDOM_SEED):
    """Oversample the minority class with SMOTE. Falls back gracefully if
    a class has too few samples for the default k_neighbors=5."""
    counts = Counter(y)
    minority_count = min(counts.values())
    if minority_count < 2 or len(counts) < 2:
        logger.warning("    SMOTE skipped: not enough samples per class.")
        return X, y

    k_neighbors = min(5, minority_count - 1)
    sm = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    X_res, y_res = sm.fit_resample(X, y)
    logger.info(f"    SMOTE: {dict(counts)} -> {dict(Counter(y_res))}")
    return X_res, y_res


def preprocess_train(X_train, y_train, use_smote, remove_outliers):
    """Apply outlier removal (first) and/or SMOTE (second) to TRAINING data
    only. Test data must always stay untouched/raw."""
    if remove_outliers:
        X_train, y_train = remove_outliers_iqr(X_train, y_train)
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)
    return X_train, y_train


def discover_folds(train_dir, test_dir):
    pattern = re.compile(r"^train_without_(.+)\.csv$")
    folds = []
    for fname in sorted(os.listdir(train_dir)):
        m = pattern.match(fname)
        if m:
            fold_id = m.group(1)
            test_path = os.path.join(test_dir, f"test_{fold_id}.csv")
            if os.path.exists(test_path):
                folds.append((fold_id, os.path.join(train_dir, fname), test_path))
            else:
                logger.warning(f"No matching test file for fold {fold_id}, skipping.")
    return folds


def load_xy(path):
    df = pd.read_csv(path)
    X = df.drop(columns=[LABEL_COL]).values
    y = df[LABEL_COL].values
    return X, y


def find_best_params(folds, n_folds, use_smote=False, remove_outliers=False):
    logger.info(f"\nPhase 1: Grid search on {n_folds} randomly selected folds …")
    hpo_folds = select_hpo_folds(folds, n_folds)
    all_best_params = []
    for idx, (fold_id, train_path, _) in enumerate(hpo_folds, 1):
        logger.info(f"  [{idx}/{len(hpo_folds)}] HPO fold: {fold_id}")
        X_train, y_train = load_xy(train_path)
        X_train, y_train = preprocess_train(X_train, y_train, use_smote, remove_outliers)
        search = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
            param_grid=HPO_PARAM_GRID,
            cv=HPO_CV,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        logger.info(f"    Best CV: {search.best_score_:.4f}  Params: {search.best_params_}")
        all_best_params.append(search.best_params_)

    param_keys = all_best_params[0].keys()
    best_params = {}
    for key in param_keys:
        values = [p[key] for p in all_best_params]
        most_common = Counter(map(str, values)).most_common(1)[0][0]
        typed_values = [p[key] for p in all_best_params if str(p[key]) == most_common]
        best_params[key] = typed_values[0]

    logger.info(f"\n  Consensus best params: {best_params}")
    return best_params, all_best_params, hpo_folds


def run_pipeline(folds, use_smote, remove_outliers, output_dir):
    """Run the full HPO + train/test LOO pipeline once, under a given
    (use_smote, remove_outliers) configuration. Writes results into a
    dedicated subfolder of output_dir so runs don't overwrite each other."""
    tag = f"smote-{use_smote}_outliers-{remove_outliers}"
    run_dir = os.path.join(output_dir, tag)
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"\n{'#'*60}\n# RUN: SMOTE={use_smote}  REMOVE_OUTLIERS={remove_outliers}\n{'#'*60}")

    best_params, hpo_params_per_fold, hpo_folds = find_best_params(
        folds, HPO_N_FOLDS, use_smote=use_smote, remove_outliers=remove_outliers
    )

    hpo_df = pd.DataFrame(hpo_params_per_fold)
    hpo_df.insert(0, "fold_id", [f[0] for f in hpo_folds])
    hpo_df.to_csv(os.path.join(run_dir, "hpo_params_per_fold.csv"), index=False)

    logger.info(f"\nPhase 2: Training all {len(folds)} folds with consensus params …")
    results = []
    all_y_true = []
    all_y_pred = []

    for idx, (fold_id, train_path, test_path) in enumerate(folds, 1):
        logger.info(f"  [{idx}/{len(folds)}] Fold: {fold_id}")
        X_train, y_train = load_xy(train_path)
        X_test,  y_test  = load_xy(test_path)

        X_train, y_train = preprocess_train(X_train, y_train, use_smote, remove_outliers)

        model = RandomForestClassifier(**best_params, random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        logger.info(f"    Test accuracy: {test_acc:.4f}")
        results.append({"fold_id": fold_id, "test_accuracy": round(test_acc, 4)})

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(run_dir, "loo_rf_results.csv"), index=False)

    logger.info(f"\n{'='*50}")
    logger.info(f"LOO Results Summary  [{tag}]")
    logger.info(f"{'='*50}")
    logger.info(f"Consensus params : {best_params}")
    logger.info(f"Mean test accuracy : {results_df['test_accuracy'].mean():.4f}")
    logger.info(f"Std  test accuracy : {results_df['test_accuracy'].std():.4f}")
    logger.info(f"Min  test accuracy : {results_df['test_accuracy'].min():.4f}")
    logger.info(f"Max  test accuracy : {results_df['test_accuracy'].max():.4f}")

    report_overall_confusion(all_y_true, all_y_pred, run_dir)
    plot_results(results_df, best_params, run_dir)

    return {
        "tag": tag,
        "use_smote": use_smote,
        "remove_outliers": remove_outliers,
        "mean_test_accuracy": results_df["test_accuracy"].mean(),
        "std_test_accuracy": results_df["test_accuracy"].std(),
        "best_params": best_params,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    folds = discover_folds(TRAIN_DIR, TEST_DIR)

    if not folds:
        logger.error("No folds found. Check TRAIN_DIR and TEST_DIR paths.")
        return

    logger.info(f"Found {len(folds)} folds.")

    if RUN_BOTH_CONDITIONS:
        configs = [(False, False), (True, True)]
    else:
        configs = [(USE_SMOTE, REMOVE_OUTLIERS)]

    summary = []
    for use_smote, remove_outliers in configs:
        summary.append(run_pipeline(folds, use_smote, remove_outliers, OUTPUT_DIR))

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "condition_comparison_summary.csv"), index=False)
    logger.info(f"\n{'='*50}\nCondition comparison\n{'='*50}\n{summary_df}")
    logger.info("\nDone.")


def report_overall_confusion(y_true, y_pred, output_dir):
    """
    Build one confusion matrix from ALL folds combined, and report
    class 0 and class 1 stats separately (counts + per-class accuracy/recall).
    Assumes binary labels {0, 1}.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = [0, 1]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    logger.info(f"\n{'='*50}")
    logger.info("Overall Confusion Matrix (all folds combined)")
    logger.info(f"{'='*50}")
    logger.info("                 Predicted 0   Predicted 1")
    logger.info(f"Actual 0   :     {tn:>10}   {fp:>11}")
    logger.info(f"Actual 1   :     {fn:>10}   {tp:>11}")


    total_0 = tn + fp
    acc_0 = tn / total_0 if total_0 else float("nan")
    logger.info(f"\nClass 0  -> total={total_0}, correct(TN)={tn}, misclassified as 1(FP)={fp}, accuracy={acc_0:.4f}")


    total_1 = fn + tp
    acc_1 = tp / total_1 if total_1 else float("nan")
    logger.info(f"Class 1  -> total={total_1}, correct(TP)={tp}, misclassified as 0(FN)={fn}, accuracy={acc_1:.4f}")

    logger.info(f"\n{classification_report(y_true, y_pred, labels=labels, digits=4)}")


    cm_df = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
    cm_df.to_csv(os.path.join(output_dir, "overall_confusion_matrix.csv"))

    per_class_df = pd.DataFrame([
        {"class": 0, "total": total_0, "correct": tn, "misclassified": fp, "accuracy": acc_0},
        {"class": 1, "total": total_1, "correct": tp, "misclassified": fn, "accuracy": acc_1},
    ])
    per_class_df.to_csv(os.path.join(output_dir, "per_class_summary.csv"), index=False)

    plot_confusion_matrix(cm, labels, output_dir)


def plot_confusion_matrix(cm, labels, output_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([f"Pred {l}" for l in labels])
    ax.set_yticklabels([f"Actual {l}" for l in labels])
    ax.set_title("Overall Confusion Matrix (all folds)")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=14, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = os.path.join(output_dir, "overall_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix plot saved → {path}")


def plot_results(results_df, best_params, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    fig.suptitle(
        f"LOO Random Forest  |  params: {best_params}",
        fontsize=10, fontweight="bold", y=0.99
    )

    folds    = results_df["fold_id"].astype(str)
    x        = np.arange(len(folds))
    test_acc = results_df["test_accuracy"].values
    mean_acc = test_acc.mean()

    ax = axes[0]
    colors = ["#E07B54" if v < mean_acc else "#4C8BB5" for v in test_acc]
    ax.bar(x, test_acc, color=colors, alpha=0.85)
    ax.axhline(mean_acc, color="black", linestyle="--", linewidth=1.2, label=f"Mean={mean_acc:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(folds, rotation=90, fontsize=7)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Per-Fold Test Accuracy (blue = above mean, orange = below)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.hist(test_acc, bins=20, color="#4C8BB5", edgecolor="white", alpha=0.85)
    ax.axvline(mean_acc, color="#E07B54", linestyle="--", linewidth=1.5, label=f"Mean={mean_acc:.3f}")
    ax.axvline(np.median(test_acc), color="#2ecc71", linestyle="--", linewidth=1.5, label=f"Median={np.median(test_acc):.3f}")
    ax.set_xlabel("Test Accuracy")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Test Accuracy Across Folds")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "loo_rf_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {path}")


if __name__ == "__main__":
    main()