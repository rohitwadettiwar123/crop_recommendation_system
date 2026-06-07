# 🌾 Explainable Crop Suitability Prediction System

> A production-quality AI web application that predicts the best crop for given soil and environmental conditions — with full explainability via SHAP.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![ML](https://img.shields.io/badge/ML-6%20Models-green)
![SHAP](https://img.shields.io/badge/XAI-SHAP-orange)

---

## 📋 Features

| Feature | Description |
|---|---|
| 🌿 Crop Prediction | Predicts best crop from 22 classes |
| 🏆 Top-3 Crops | Shows top-3 recommendations with confidence |
| 📊 Suitability Score | 0–100 gauge (Excellent / Very Good / Good / Moderate / Poor) |
| 🧠 SHAP Explainability | Local & Global explanations |
| 📈 Feature Importance | Visual ranking of which features drove prediction |
| 📉 Interactive Charts | Plotly dashboards for EDA and model insights |
| 🔴 Real-time Prediction | Instant results from slider inputs |

---

## 🗂️ Project Structure

```
crop_system/
├── data/
│   └── Crop_recommendation.csv        ← Place your dataset here
├── models/                            ← Auto-created after training
│   ├── crop_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── model_metadata.json
├── src/
│   ├── preprocessing.py               ← Data cleaning & feature engineering
│   ├── train_model.py                 ← Model comparison & selection
│   ├── predictor.py                   ← Prediction + reasoning engine
│   └── explainability.py             ← SHAP wrappers + Plotly charts
├── notebooks/
│   └── EDA.ipynb                      ← Exploratory data analysis notebook
├── app.py                             ← Streamlit web application (main)
├── requirements.txt
├── render.yaml                        ← Render.com deployment config
└── README.md
```

---

## 🚀 Quick Start (Local)

### 1. Clone and setup
```bash
git clone <your-repo-url>
cd crop_system
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add the dataset
Place `Crop_recommendation.csv` inside the `data/` folder.

Dataset columns: `N, P, K, temperature, humidity, ph, rainfall, label`

### 3. Run the app
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 4. Train the model
In the app, navigate to **Model Training** → click **Start Training**.
The app will:
- Clean and preprocess data
- Compare 6 ML models
- Select the best model automatically
- Save trained model to `models/`

---

## 🤖 ML Models Compared

| Model | Typical Accuracy |
|---|---|
| Random Forest | ~99% |
| XGBoost | ~99% |
| LightGBM | ~99% |
| Decision Tree | ~98% |
| KNN | ~97% |
| Logistic Regression | ~95% |

Best model is selected automatically by **weighted F1-score**.

---

## 🧠 Explainability (SHAP)

- **Global**: Mean absolute SHAP values across the dataset — which features matter most overall
- **Local**: Per-prediction SHAP waterfall — why *this specific* crop was recommended for *these* inputs
- **Reasoning**: Human-readable bullet points comparing inputs to ideal ranges

---

## 📦 Dataset

- **Name**: Crop Recommendation Dataset
- **Rows**: 2,200
- **Classes**: 22 crops (rice, maize, coffee, cotton, banana, mango, ...)
- **Features**: Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH, Rainfall

---

## ☁️ Deploy on Render

1. Push repository to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
5. Add environment variable: `PYTHON_VERSION = 3.10.12`
6. Click **Deploy**

> ⚠️ **Important**: Either pre-train and commit the `models/` folder, or train via the UI after first deploy.

---

## 📊 Suitability Score Legend

| Score | Label |
|---|---|
| ≥ 90% | 🟢 Excellent |
| 80–90% | 🟡 Very Good |
| 70–80% | 🟡 Good |
| 60–70% | 🟠 Moderate |
| < 60% | 🔴 Poor |

---


