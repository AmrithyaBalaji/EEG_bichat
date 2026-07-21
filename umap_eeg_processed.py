import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt

TRAIN_PATH = r"C:\abalaji\bichat\PROCESSED_LOO\0\train_without_1214.csv"
TEST_PATH = r"C:\abalaji\bichat\PROCESSED_LOO\1\test_1214.csv"


def load(path):
    raw = pd.read_csv(path, header=None)
    first_row_numeric = pd.to_numeric(raw.iloc[0], errors="coerce").notna().all()
    if not first_row_numeric:
        raw = pd.read_csv(path)
    arr = raw.values.astype(float)
    return arr[:, :-1], arr[:, -1]


X_train, y_train = load(TRAIN_PATH)
X_test, y_test = load(TEST_PATH)

print("train shape:", X_train.shape, "test shape:", X_test.shape)
print("train label counts:", dict(zip(*np.unique(y_train, return_counts=True))))
print("test label counts:", dict(zip(*np.unique(y_test, return_counts=True))))

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean", random_state=42)
embedding_train = reducer.fit_transform(X_train)
embedding_test = reducer.transform(X_test)

fig, ax = plt.subplots(figsize=(8, 6))
for label in np.unique(y_train):
    mask = y_train == label
    ax.scatter(embedding_train[mask, 0], embedding_train[mask, 1], s=8, alpha=0.4, label=f"train label {int(label)}")
for label in np.unique(y_test):
    mask = y_test == label
    ax.scatter(embedding_test[mask, 0], embedding_test[mask, 1], s=40, alpha=0.9, marker="x", label=f"test (1214) label {int(label)}")
ax.legend()
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title("UMAP: fit on train_without_1214, patient 1214 projected as test")
plt.tight_layout()
plt.savefig("umap_train_test_1214.png", dpi=200)
plt.show()