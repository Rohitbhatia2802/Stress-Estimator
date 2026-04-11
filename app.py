"""
Stress Level Estimator — Streamlit App
=======================================
Loads trained sklearn models + preprocessor.
Accepts uploaded image or webcam capture.
Shows emotion predictions, consensus, stress level, and confidence chart.
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODELS_DIR = "models"

EMOTION_LABELS = {
    0: "Angry", 1: "Disgust", 2: "Fear",
    3: "Happy", 4: "Sad",    5: "Surprise", 6: "Neutral"
}

# Emotion → Stress mapping (heuristic)
STRESS_MAP = {
    "Angry":   ("High",   "#E85D24"),
    "Disgust": ("Medium", "#F2A623"),
    "Fear":    ("High",   "#E85D24"),
    "Happy":   ("Low",    "#3B8BD4"),
    "Sad":     ("Medium", "#F2A623"),
    "Surprise":("Medium", "#F2A623"),
    "Neutral": ("Low",    "#3B8BD4"),
}

STRESS_EMOJI = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

MODEL_FILES = {
    "Decision Tree": "dt_model.pkl",
    "Random Forest": "rf_model.pkl",
    "SVM":           "svm_model.pkl",
}

# ─────────────────────────────────────────────
# CACHED LOADERS (run once per session)
# ─────────────────────────────────────────────
@st.cache_resource
def load_preprocessor():
    path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource
def load_models():
    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


@st.cache_resource
def load_summary():
    path = os.path.join(MODELS_DIR, "model_summary.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image → 48×48 grayscale → normalised → flat (1, 2304)"""
    img = pil_image.convert("L")           # grayscale
    img = img.resize((48, 48), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)  # shape (48, 48)
    arr = arr.flatten().reshape(1, -1)     # shape (1, 2304)
    return arr


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def run_inference(raw_features, preprocessor, models):
    """
    Transform raw pixel features and run all models.
    Returns dict with emotion string and probabilities per model.
    """
    X_pca = preprocessor.transform(raw_features)   # (1, 80)

    predictions = {}
    for name, model in models.items():
        pred_idx = model.predict(X_pca)[0]
        emotion  = EMOTION_LABELS[pred_idx]

        # Probabilities (SVC has probability=True; DT/RF have predict_proba)
        try:
            probs = model.predict_proba(X_pca)[0]   # shape (7,)
        except Exception:
            probs = np.zeros(7)
            probs[pred_idx] = 1.0

        predictions[name] = {
            "emotion":     emotion,
            "emotion_idx": pred_idx,
            "probs":       probs,
        }

    return predictions


def consensus_emotion(predictions):
    """Majority vote across models."""
    votes = [p["emotion"] for p in predictions.values()]
    return max(set(votes), key=votes.count)


# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
def plot_confidence(predictions):
    """Return a matplotlib figure with side-by-side confidence bars."""
    emotion_names = [EMOTION_LABELS[i] for i in range(7)]
    n_models = len(predictions)

    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 3.5),
                             facecolor="#0E1117")
    if n_models == 1:
        axes = [axes]

    colors_map = {
        "Angry": "#E85D24", "Disgust": "#F2A623", "Fear": "#E85D24",
        "Happy": "#3B8BD4", "Sad":    "#888780",  "Surprise": "#F2A623",
        "Neutral": "#3B8BD4"
    }

    for ax, (name, pred) in zip(axes, predictions.items()):
        probs  = pred["probs"] * 100
        colors = [colors_map.get(e, "#888780") for e in emotion_names]
        bars   = ax.barh(emotion_names, probs, color=colors, height=0.55)

        # Highlight the top bar
        top_idx = np.argmax(probs)
        bars[top_idx].set_edgecolor("white")
        bars[top_idx].set_linewidth(1.5)

        ax.set_facecolor("#0E1117")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Confidence (%)", color="#CCCCCC", fontsize=9)
        ax.set_title(name, color="white", fontsize=10, fontweight="bold")
        ax.tick_params(colors="#CCCCCC", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#444444")
        ax.spines["left"].set_color("#444444")

        # Value labels on bars
        for i, (bar, p) in enumerate(zip(bars, probs)):
            if p > 2:
                ax.text(p + 1, bar.get_y() + bar.get_height() / 2,
                        f"{p:.1f}%", va="center", color="#CCCCCC", fontsize=8)

    plt.tight_layout(pad=1.5)
    return fig


def plot_accuracy_table(summary):
    """Return a styled dataframe for display."""
    rows = []
    for name, stats in summary.items():
        rows.append({
            "Model":         name,
            "Val accuracy":  f"{stats['val_acc']:.2%}",
            "Val F1-macro":  f"{stats['val_f1']:.4f}",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# UI LAYOUT
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Stress Level Estimator",
        page_icon="🧠",
        layout="wide"
    )

    # ── Header ──────────────────────────────
    st.title("🧠 Stress Level Estimator")
    st.markdown(
        "Upload a facial image or use your webcam. "
        "Three ML models predict the expressed emotion, "
        "which is mapped to a **stress level**."
    )
    st.divider()

    # ── Check models are loaded ──────────────
    preprocessor = load_preprocessor()
    models       = load_models()
    summary      = load_summary()

    if preprocessor is None or len(models) == 0:
        st.error(
            "**Models not found.** Please run `python train.py` first "
            "to train the models and save them in the `models/` folder."
        )
        st.code("python train.py", language="bash")
        st.stop()

    st.success(f"✅ Loaded {len(models)} models: {', '.join(models.keys())}")

    # ── Sidebar — model accuracy table ──────
    with st.sidebar:
        st.header("📊 Model performance")
        if summary:
            df_summary = plot_accuracy_table(summary)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
        else:
            st.info("Run training to see accuracy stats.")

        st.header("🗺️ Stress map")
        for emotion, (level, color) in STRESS_MAP.items():
            emoji = STRESS_EMOJI[level]
            st.markdown(
                f"<span style='color:{color}'><b>{emotion}</b></span> → "
                f"{emoji} {level}",
                unsafe_allow_html=True
            )

        st.header("ℹ️ How it works")
        st.markdown(
            "1. Image → 48×48 grayscale\n"
            "2. Pixels flattened → 2304 features\n"
            "3. StandardScaler → PCA (80 components)\n"
            "4. All 3 models predict emotion\n"
            "5. Majority vote → consensus emotion\n"
            "6. Emotion mapped to stress level"
        )

    # ── Image input ─────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📸 Input image")
        input_method = st.radio(
            "Choose input method",
            ["Upload image", "Webcam capture"],
            horizontal=True
        )

        pil_image = None

        if input_method == "Upload image":
            uploaded = st.file_uploader(
                "Upload a face photo (JPG, PNG, WEBP)",
                type=["jpg", "jpeg", "png", "webp"]
            )
            if uploaded:
                pil_image = Image.open(uploaded)

        else:  # Webcam
            camera_img = st.camera_input("Take a photo")
            if camera_img:
                pil_image = Image.open(camera_img)

        if pil_image:
            st.image(pil_image, caption="Input image", use_container_width=True)

            # Show the 48×48 preview
            preview = pil_image.convert("L").resize((48, 48), Image.LANCZOS)
            preview_big = preview.resize((144, 144), Image.NEAREST)
            st.image(preview_big, caption="48×48 grayscale input to model", width=150)

    # ── Results ─────────────────────────────
    with col2:
        st.subheader("🔍 Predictions")

        if pil_image is None:
            st.info("👆 Upload or capture an image to see predictions.")
        else:
            with st.spinner("Running inference on all 3 models..."):
                try:
                    raw = preprocess_image(pil_image)
                    predictions = run_inference(raw, preprocessor, models)
                    consensus   = consensus_emotion(predictions)
                    stress_level, stress_color = STRESS_MAP[consensus]
                    stress_emoji = STRESS_EMOJI[stress_level]

                    # ── Consensus banner ────────────────
                    st.markdown(
                        f"""
                        <div style='
                            background:{stress_color}22;
                            border:2px solid {stress_color};
                            border-radius:12px;
                            padding:16px 20px;
                            margin-bottom:16px;
                        '>
                            <div style='font-size:13px;color:#aaa;margin-bottom:4px'>
                                Consensus emotion (majority vote)
                            </div>
                            <div style='font-size:22px;font-weight:700;color:{stress_color}'>
                                {consensus}
                            </div>
                            <div style='font-size:18px;margin-top:6px'>
                                {stress_emoji} Stress level: <b>{stress_level}</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # ── Per-model results ────────────────
                    st.markdown("**Per-model predictions:**")
                    cols = st.columns(len(predictions))
                    for col, (name, pred) in zip(cols, predictions.items()):
                        emo   = pred["emotion"]
                        conf  = pred["probs"].max() * 100
                        s_lvl, s_col = STRESS_MAP[emo]
                        col.metric(
                            label=name,
                            value=emo,
                            delta=f"{conf:.1f}% confidence"
                        )

                except Exception as e:
                    st.error(f"Inference failed: {e}")
                    st.stop()

    # ── Confidence chart ────────────────────
    if pil_image is not None:
        st.divider()
        st.subheader("📈 Confidence probabilities per model")
        fig = plot_confidence(predictions)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── Raw probabilities table ──────────
        with st.expander("🔢 Raw probability table"):
            emotion_names = [EMOTION_LABELS[i] for i in range(7)]
            prob_data = {}
            for name, pred in predictions.items():
                prob_data[name] = [f"{p:.3f}" for p in pred["probs"]]
            df_probs = pd.DataFrame(prob_data, index=emotion_names)
            st.dataframe(df_probs, use_container_width=True)

    st.divider()
    st.caption(
        "Built with scikit-learn · FER2013 dataset · "
        "Models: Decision Tree, Random Forest, SVM (RBF kernel)"
    )


if __name__ == "__main__":
    main()
