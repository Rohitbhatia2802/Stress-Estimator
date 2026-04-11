"""
Stress Level Estimator — Training Script
FER2013 Dataset | scikit-learn pipeline
=========================================
Models: Decision Tree, Random Forest, SVM (RBF)
Preprocessing: Flatten → StandardScaler → PCA(80)
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH   = "data/fer2013.csv"
MODELS_DIR  = "models"
N_PCA       = 80           # PCA components (captures ~85% variance)
IMG_SIZE    = 48 * 48      # 2304 raw features after flatten
EMOTION_LABELS = {
    0: "Angry", 1: "Disgust", 2: "Fear",
    3: "Happy", 4: "Sad",    5: "Surprise", 6: "Neutral"
}

os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1 — Load FER2013
# ─────────────────────────────────────────────
def load_fer2013(path: str):
    """
    Load FER2013 CSV. Returns train and val splits.
    Columns expected: emotion (int), pixels (space-sep string), Usage (str)
    """
    print(f"\n[1/6] Loading dataset from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Make sure fer2013.csv is inside the 'data/' folder."
        )

    df = pd.read_csv(path)
    print(f"      Total rows: {len(df)}")
    print(f"      Columns   : {list(df.columns)}")
    print(f"      Usage split:\n{df['Usage'].value_counts().to_string()}")

    # Split by Usage column (FER2013 built-in split)
    train_df = df[df["Usage"] == "Training"].copy()
    val_df   = df[df["Usage"] == "PublicTest"].copy()

    print(f"\n      Train size: {len(train_df)} | Val size: {len(val_df)}")
    return train_df, val_df


def parse_pixels(df: pd.DataFrame):
    """Convert space-separated pixel string → numpy array (n_samples, 2304)"""
    pixels = np.array([
        list(map(int, row.split()))
        for row in df["pixels"]
    ], dtype=np.float32)
    labels = df["emotion"].values
    return pixels, labels


# ─────────────────────────────────────────────
# STEP 2 — Preprocess (NO leakage)
# ─────────────────────────────────────────────
def build_preprocessor(X_train: np.ndarray):
    """
    Fit StandardScaler + PCA on TRAINING data ONLY.
    Returns fitted pipeline (to be applied to val/test later).
    """
    print(f"\n[2/6] Building preprocessor: StandardScaler -> PCA({N_PCA})")
    preprocessor = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=N_PCA, random_state=42))
    ])
    preprocessor.fit(X_train)
    explained = preprocessor.named_steps["pca"].explained_variance_ratio_.sum()
    print(f"      PCA fitted. Explained variance: {explained:.2%}")
    return preprocessor


# ─────────────────────────────────────────────
# STEP 3 — Define models
# ─────────────────────────────────────────────
def get_models():
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=20,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=25,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),
        "SVM": SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,  # needed for confidence scores in Streamlit
            random_state=42,
            cache_size=500
        )
    }


# ─────────────────────────────────────────────
# STEP 4 — Train & evaluate
# ─────────────────────────────────────────────
def train_and_evaluate(models, X_train_pca, y_train, X_val_pca, y_val):
    results = {}
    print(f"\n[3/6] Training {len(models)} models...\n")

    for name, model in models.items():
        print(f"  -- Training: {name}")
        t0 = time.time()
        model.fit(X_train_pca, y_train)
        elapsed = time.time() - t0

        # Predictions
        y_pred_train = model.predict(X_train_pca)
        y_pred_val   = model.predict(X_val_pca)

        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc   = accuracy_score(y_val,   y_pred_val)
        val_f1    = f1_score(y_val, y_pred_val, average="macro", zero_division=0)
        cm        = confusion_matrix(y_val, y_pred_val)

        results[name] = {
            "model":     model,
            "train_acc": train_acc,
            "val_acc":   val_acc,
            "val_f1":    val_f1,
            "cm":        cm,
            "elapsed":   elapsed
        }

        print(f"     Train acc : {train_acc:.4f}")
        print(f"     Val acc   : {val_acc:.4f}")
        print(f"     Val F1-mac: {val_f1:.4f}")
        print(f"     Time      : {elapsed:.1f}s\n")

        print("  Classification report (val):")
        report = classification_report(
            y_val, y_pred_val,
            target_names=[EMOTION_LABELS[i] for i in range(7)],
            zero_division=0
        )
        print(report)

    return results


# ─────────────────────────────────────────────
# STEP 5 — Save confusion matrix plot
# ─────────────────────────────────────────────
def save_confusion_matrices(results):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        print("[5/6] Saving confusion matrix plots...")
        emotion_names = [EMOTION_LABELS[i] for i in range(7)]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Confusion Matrices - Validation Set", fontsize=14)

        for ax, (name, res) in zip(axes, results.items()):
            sns.heatmap(
                res["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=emotion_names, yticklabels=emotion_names,
                ax=ax, cbar=False
            )
            ax.set_title(f"{name}\nAcc={res['val_acc']:.3f} F1={res['val_f1']:.3f}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plot_path = os.path.join(MODELS_DIR, "confusion_matrices.png")
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"      Saved → {plot_path}")
    except Exception as e:
        print(f"      (Plot skipped: {e})")


# ─────────────────────────────────────────────
# STEP 6 — Save models + preprocessor
# ─────────────────────────────────────────────
def save_artifacts(preprocessor, results):
    print("\n[6/6] Saving models and preprocessor...")

    # Save preprocessor (scaler + PCA bundled)
    pre_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    joblib.dump(preprocessor, pre_path)
    print(f"      Saved → {pre_path}")

    # Save each model
    filename_map = {
        "Decision Tree": "dt_model.pkl",
        "Random Forest": "rf_model.pkl",
        "SVM":           "svm_model.pkl"
    }
    # Save accuracy summary for Streamlit
    summary = {}
    for name, res in results.items():
        fname = filename_map[name]
        fpath = os.path.join(MODELS_DIR, fname)
        joblib.dump(res["model"], fpath)
        summary[name] = {
            "val_acc": round(res["val_acc"], 4),
            "val_f1":  round(res["val_f1"],  4)
        }
        print(f"      Saved → {fpath}")

    # Save summary as a small joblib dict (used by Streamlit for comparison table)
    summary_path = os.path.join(MODELS_DIR, "model_summary.pkl")
    joblib.dump(summary, summary_path)
    print(f"      Saved → {summary_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  STRESS LEVEL ESTIMATOR - TRAINING PIPELINE")
    print("=" * 55)

    # 1. Load data
    train_df, val_df = load_fer2013(DATA_PATH)

    # 2. Parse pixels → numpy arrays
    print("\n[2a/6] Parsing pixel strings...")
    X_train_raw, y_train = parse_pixels(train_df)
    X_val_raw,   y_val   = parse_pixels(val_df)
    print(f"       X_train shape: {X_train_raw.shape}")
    print(f"       X_val shape  : {X_val_raw.shape}")

    # 3. Preprocess — fit ONLY on train, transform both
    print("\n[2b/6] Fitting preprocessor on training data only...")
    preprocessor = build_preprocessor(X_train_raw)

    X_train_pca = preprocessor.transform(X_train_raw)
    X_val_pca   = preprocessor.transform(X_val_raw)
    print(f"       After PCA — X_train: {X_train_pca.shape}, X_val: {X_val_pca.shape}")

    # 4. Train all models
    models  = get_models()
    results = train_and_evaluate(models, X_train_pca, y_train, X_val_pca, y_val)

    # 5. Plots
    save_confusion_matrices(results)

    # 6. Save everything
    save_artifacts(preprocessor, results)

    # Summary table
    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE - SUMMARY")
    print("=" * 55)
    print(f"  {'Model':<18} {'Val Acc':>8} {'Val F1':>8} {'Time':>8}")
    print("  " + "-" * 44)
    for name, res in results.items():
        print(f"  {name:<18} {res['val_acc']:>8.4f} {res['val_f1']:>8.4f} {res['elapsed']:>6.1f}s")
    print("\n  All files saved to: models/")
    print("  Run 'streamlit run app.py' to launch the UI.\n")


if __name__ == "__main__":
    main()
