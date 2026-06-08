"""
explainability.py
=================
SHAP-based explainability module for the Crop Suitability Prediction System.
Provides local (per-prediction) and global (dataset-wide) explanations.
"""

import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

FEATURE_COLS   = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
FEATURE_LABELS = ['Nitrogen','Phosphorus','Potassium','Temperature','Humidity','pH','Rainfall']


def get_shap_explainer(model, X_train_sample: np.ndarray, model_name: str = ""):
    """
    Build a SHAP explainer appropriate for the model type.

    Uses TreeExplainer for tree-based models, LinearExplainer for Logistic Regression,
    and KernelExplainer as fallback.
    """
    tree_based = ["Random Forest","XGBoost","LightGBM","Decision Tree"]
    linear_based = ["Logistic Regression"]

    try:
        if any(t in model_name for t in tree_based):
            explainer = shap.TreeExplainer(model)
        elif any(t in model_name for t in linear_based):
            explainer = shap.LinearExplainer(model, X_train_sample)
        else:
            background = shap.sample(X_train_sample, min(100, len(X_train_sample)))
            explainer  = shap.KernelExplainer(model.predict_proba, background)
    except Exception:
        background = shap.sample(X_train_sample, min(50, len(X_train_sample)))
        explainer  = shap.KernelExplainer(model.predict_proba, background)

    return explainer


def compute_local_shap(explainer, input_scaled: np.ndarray,
                        predicted_class_idx: int, class_names: list) -> dict:
    """
    Compute SHAP values for a single prediction (local explanation).

    Returns dict with shap_values array and base_value.
    """
    try:
        shap_vals = explainer.shap_values(input_scaled)

        # Handle multi-class output
        if isinstance(shap_vals, list):
            # shap_vals[class_idx] for multiclass TreeExplainer
            if predicted_class_idx < len(shap_vals):
                local_shap = shap_vals[predicted_class_idx][0]
            else:
                local_shap = shap_vals[0][0]
            base_val = explainer.expected_value[predicted_class_idx] \
                       if hasattr(explainer.expected_value, '__len__') \
                       else explainer.expected_value
        elif len(shap_vals.shape) == 3:
            # (1, n_features, n_classes)
            local_shap = shap_vals[0, :, predicted_class_idx]
            base_val   = float(explainer.expected_value[predicted_class_idx]) \
                         if hasattr(explainer.expected_value, '__len__') \
                         else float(explainer.expected_value)
        else:
            local_shap = shap_vals[0]
            base_val   = float(explainer.expected_value)

        return {"shap_values": local_shap, "base_value": float(base_val)}

    except Exception as e:
        # Fallback: zero SHAP values
        return {"shap_values": np.zeros(len(FEATURE_COLS)), "base_value": 0.0}


def compute_global_shap(explainer, X_scaled: np.ndarray,
                         model_name: str = "") -> np.ndarray:
    """
    Compute mean absolute SHAP values across dataset for global importance.

    Returns array of shape (n_features,) — mean |SHAP| per feature.
    """
    try:
        sample = X_scaled[:min(300, len(X_scaled))]
        shap_vals = explainer.shap_values(sample)

        if isinstance(shap_vals, list):
            # Average across classes
            abs_vals = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
        elif len(shap_vals.shape) == 3:
            abs_vals = np.mean(np.abs(shap_vals), axis=(0, 2))
        else:
            abs_vals = np.abs(shap_vals)

        mean_abs = np.mean(abs_vals, axis=0)
        return mean_abs

    except Exception:
        # Fallback to model feature importances
        if hasattr(explainer.model, 'feature_importances_'):
            return explainer.model.feature_importances_
        return np.ones(len(FEATURE_COLS)) / len(FEATURE_COLS)


# ───────────────────────────────────────────
# PLOTLY CHARTS
# ───────────────────────────────────────────

def plot_local_explanation(shap_values: np.ndarray, crop_name: str) -> go.Figure:
    """Waterfall-style bar chart showing feature contributions for a prediction."""
    labels = FEATURE_LABELS
    colors = ['#00C853' if v >= 0 else '#D50000' for v in shap_values]
    sorted_idx = np.argsort(np.abs(shap_values))[::-1]

    fig = go.Figure(go.Bar(
        x=[shap_values[i] for i in sorted_idx],
        y=[labels[i] for i in sorted_idx],
        orientation='h',
        marker_color=[colors[i] for i in sorted_idx],
        text=[f"{shap_values[i]:+.4f}" for i in sorted_idx],
        textposition='outside',
    ))
    fig.update_layout(
        title=f"Local SHAP Explanation — {crop_name.title()}",
        xaxis_title="SHAP Value (impact on prediction)",
        yaxis_title="Feature",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        title_font_size=16,
    )
    return fig


def plot_global_importance(mean_shap: np.ndarray) -> go.Figure:
    """Horizontal bar chart of mean |SHAP| values (global feature importance)."""
    sorted_idx  = np.argsort(mean_shap)
    sorted_vals = mean_shap[sorted_idx]
    sorted_labs = [FEATURE_LABELS[i] for i in sorted_idx]

    fig = go.Figure(go.Bar(
        x=sorted_vals,
        y=sorted_labs,
        orientation='h',
        marker=dict(
            color=sorted_vals,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=[f"{v:.4f}" for v in sorted_vals],
        textposition='outside',
    ))
    fig.update_layout(
        title="Global Feature Importance (Mean |SHAP|)",
        xaxis_title="Mean |SHAP Value|",
        height=420,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        title_font_size=16,
    )
    return fig


def plot_feature_contribution_bar(contributions: dict, crop_name: str) -> go.Figure:
    """Percentage contribution bar chart from reasoning module."""
    labels = list(contributions.keys())
    values = list(contributions.values())
    colors = px.colors.qualitative.Set2[:len(labels)]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition='outside',
    ))
    fig.update_layout(
        title=f"Feature Contributions for {crop_name.title()}",
        yaxis_title="Contribution (%)",
        height=380,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        title_font_size=16,
    )
    return fig
