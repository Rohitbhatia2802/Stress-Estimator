# Stress Level Estimator 🧠

This project utilizes the **FER2013 dataset** and **scikit-learn** to predict emotion from facial expressions and map them to heuristic stress levels.

## Features
- **Three-Model Consensus:** Uses **Decision Tree**, **Random Forest**, and **SVM** models for majority voting.
- **Preprocessing Pipeline:** Includes Grayscale conversion, StandardScaling, and **PCA (80 components)** for dimensionality reduction.
- **Streamlit UI:** Interactive web interface for uploading images or using a webcam for real-time inference.
- **Confidence Metrics:** Visualizes confidence scores per model and provides a detailed probability breakdown.

## Folder Structure
```
stress_estimator/
│
├── data/
│   └── fer2013.csv            ← Your downloaded dataset (Kaggle FER2013)
│
├── models/                    ← Auto-created and populated by train.py
│   ├── preprocessor.pkl
│   ├── dt_model.pkl
│   ├── rf_model.pkl
│   ├── svm_model.pkl
│   └── model_summary.pkl
│
├── train.py                   ← Model training and evaluation script
├── app.py                     ← Streamlit interactive web application
├── requirements.txt           ← Project dependencies
└── README.md                  ← This file
```

## Getting Started

### 1. Install Dependencies
Ensure you have the required libraries installed:
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Place the `fer2013.csv` file in the `data/` directory.

### 3. Train Models
Run the training script to generate the preprocessor and models:
```bash
python train.py
```

### 4. Launch Application
Start the Streamlit UI to begin estimating stress levels:
```bash
streamlit run app.py
```

## Disclaimer
⚠️ This tool is a research demo. Stress estimation from facial expression is heuristic and **not a clinical assessment**. Results are for instructional purposes only.
