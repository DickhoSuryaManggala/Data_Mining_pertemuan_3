import os
import numpy as np
import random
import pandas as pd

SEED = 0
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1) Load dataset

CSV_PATH = "dataset_perusahaan_klien_telat_20.csv"  
dataset = pd.read_csv(CSV_PATH)

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]  # Status_Pembayaran (Tepat/Telat)

# 2) Kolom numerik & kategorikal

categorical_cols = ["Jenis_Kelamin", "Status_Pernikahan"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

# 3) Pipeline preprocessing

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Encode target (Telat/Tepat -> angka)
le = LabelEncoder()
y_enc = le.fit_transform(y) 


# 4) Train/Test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=SEED, stratify=y_enc
)

# 5) Fit & transform

X_train_proc = preprocess.fit_transform(X_train)
X_test_proc  = preprocess.transform(X_test)

# Densify bila sparse
X_train_proc = X_train_proc.toarray() if hasattr(X_train_proc, "toarray") else X_train_proc
X_test_proc  = X_test_proc.toarray()  if hasattr(X_test_proc, "toarray")  else X_test_proc

# Nama fitur setelah OneHot
ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = list(ohe.get_feature_names_out(categorical_cols))
feature_names = numeric_cols + cat_feature_names

print("Fitur numerik:", numeric_cols)
print("Fitur kategori (OneHot):", cat_feature_names)
print("Shape X_train ->", X_train_proc.shape, "| Shape X_test ->", X_test_proc.shape)
print("Distribusi label train:", np.bincount(y_train), "(urut label:", list(le.classes_), ")")

# 6) Modeling & evaluasi

# Logistic Regression
log_reg = LogisticRegression(random_state=SEED, max_iter=1000)
log_reg.fit(X_train_proc, y_train)
y_pred_log = log_reg.predict(X_test_proc)

print("\n=== Logistic Regression ===")
print("Akurasi:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log, target_names=le.classes_))

# Decision Tree
tree = DecisionTreeClassifier(random_state=SEED, max_depth=3)
tree.fit(X_train_proc, y_train)
y_pred_tree = tree.predict(X_test_proc)

print("\n=== Decision Tree ===")
print("Akurasi:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree, target_names=le.classes_))


# 7) Visualisasi -> simpan PNG (tanpa plt.show())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ConfusionMatrixDisplay.from_estimator(
    log_reg, X_test_proc, y_test,
    display_labels=le.classes_, cmap="Blues", values_format="d", ax=axes[0]
)
axes[0].set_title("Confusion Matrix - Logistic Regression")

ConfusionMatrixDisplay.from_estimator(
    tree, X_test_proc, y_test,
    display_labels=le.classes_, cmap="Greens", values_format="d", ax=axes[1]
)
axes[1].set_title("Confusion Matrix - Decision Tree")

plt.tight_layout()
plt.savefig("confusion_matrix_models.png", dpi=300)
print("✅ Disimpan: confusion_matrix_models.png")

# Perbandingan akurasi
akurasi = {
    "Logistic Regression": accuracy_score(y_test, y_pred_log),
    "Decision Tree": accuracy_score(y_test, y_pred_tree)
}

plt.figure(figsize=(6, 4))
bars = plt.bar(list(akurasi.keys()), list(akurasi.values()), color=["skyblue", "lightgreen"])
plt.title("Perbandingan Akurasi Model")
plt.ylabel("Akurasi")
plt.ylim(0, 1)
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("perbandingan_akurasi.png", dpi=300)
print("✅ Disimpan: perbandingan_akurasi.png")

# 8) Simpan artifacts (CSV, NPY, Joblib)
import joblib

# Split mentah (sebelum transform)
pd.DataFrame(X_train, columns=X.columns).to_csv("X_train.csv", index=False)
pd.DataFrame(X_test, columns=X.columns).to_csv("X_test.csv", index=False)
pd.DataFrame(y_train, columns=["y_train"]).to_csv("y_train.csv", index=False)
pd.DataFrame(y_test, columns=["y_test"]).to_csv("y_test.csv", index=False)

# Setelah preprocessing
pd.DataFrame(X_train_proc, columns=feature_names).to_csv("X_train_processed.csv", index=False)
pd.DataFrame(X_test_proc, columns=feature_names).to_csv("X_test_processed.csv", index=False)
pd.DataFrame(y_train, columns=["y_train_encoded"]).to_csv("y_train_encoded.csv", index=False)
pd.DataFrame(y_test, columns=["y_test_encoded"]).to_csv("y_test_encoded.csv", index=False)

# Numpy arrays (cepat untuk reuse)
np.save("X_train.npy", X_train_proc)
np.save("X_test.npy", X_test_proc)

# Simpan preprocessor & salah satu model (contoh: Decision Tree)
joblib.dump(preprocess, "preprocessor.joblib")
joblib.dump(tree, "quick_model.joblib")
print("✅ Semua file CSV/NPY/Joblib berhasil disimpan!")
