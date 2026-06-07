"""
predictor.py
============
Real-time crop prediction with confidence scores, top-3 crops,
suitability score, and human-readable reasoning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Crop-specific ideal ranges for reasoning generation
CROP_IDEAL_RANGES = {
    'rice':       {'N':(60,100), 'P':(30,60),  'K':(30,60),  'temperature':(20,27), 'humidity':(80,90), 'ph':(5.5,7.0), 'rainfall':(150,300)},
    'maize':      {'N':(50,90),  'P':(30,60),  'K':(20,50),  'temperature':(18,27), 'humidity':(55,75), 'ph':(5.5,7.5), 'rainfall':(50,120)},
    'chickpea':   {'N':(0,40),   'P':(40,80),  'K':(20,40),  'temperature':(15,25), 'humidity':(15,35), 'ph':(6.0,8.0), 'rainfall':(30,100)},
    'kidneybeans':{'N':(0,40),   'P':(60,120), 'K':(15,40),  'temperature':(15,25), 'humidity':(18,28), 'ph':(5.5,7.0), 'rainfall':(30,100)},
    'pigeonpeas': {'N':(0,40),   'P':(40,80),  'K':(15,40),  'temperature':(18,35), 'humidity':(30,55), 'ph':(5.0,7.0), 'rainfall':(60,150)},
    'mothbeans':  {'N':(0,25),   'P':(30,60),  'K':(20,40),  'temperature':(25,38), 'humidity':(25,50), 'ph':(4.5,7.5), 'rainfall':(30,80)},
    'mungbean':   {'N':(0,40),   'P':(40,80),  'K':(15,40),  'temperature':(25,35), 'humidity':(60,85), 'ph':(6.2,7.2), 'rainfall':(60,130)},
    'blackgram':  {'N':(0,40),   'P':(40,80),  'K':(15,40),  'temperature':(25,35), 'humidity':(60,80), 'ph':(5.0,7.5), 'rainfall':(60,120)},
    'lentil':     {'N':(0,30),   'P':(30,60),  'K':(10,25),  'temperature':(15,25), 'humidity':(60,75), 'ph':(6.0,8.0), 'rainfall':(30,100)},
    'pomegranate':{'N':(0,10),   'P':(10,40),  'K':(40,80),  'temperature':(18,38), 'humidity':(85,95), 'ph':(5.5,7.5), 'rainfall':(50,100)},
    'banana':     {'N':(80,120), 'P':(50,80),  'K':(40,60),  'temperature':(20,35), 'humidity':(75,90), 'ph':(5.5,7.0), 'rainfall':(100,200)},
    'mango':      {'N':(0,20),   'P':(10,40),  'K':(30,60),  'temperature':(24,35), 'humidity':(45,70), 'ph':(5.5,7.5), 'rainfall':(50,150)},
    'grapes':     {'N':(0,20),   'P':(10,40),  'K':(30,80),  'temperature':(8,38),  'humidity':(55,80), 'ph':(5.5,7.0), 'rainfall':(30,80)},
    'watermelon': {'N':(80,120), 'P':(10,40),  'K':(40,60),  'temperature':(24,35), 'humidity':(80,95), 'ph':(6.0,7.0), 'rainfall':(40,80)},
    'muskmelon':  {'N':(80,120), 'P':(10,40),  'K':(40,60),  'temperature':(28,38), 'humidity':(85,95), 'ph':(6.0,7.0), 'rainfall':(20,50)},
    'apple':      {'N':(0,20),   'P':(100,150),'K':(150,200),'temperature':(0,24),  'humidity':(80,95), 'ph':(5.5,7.0), 'rainfall':(50,120)},
    'orange':     {'N':(0,20),   'P':(10,40),  'K':(5,20),   'temperature':(10,35), 'humidity':(85,95), 'ph':(6.0,7.5), 'rainfall':(60,110)},
    'papaya':     {'N':(40,60),  'P':(30,60),  'K':(40,60),  'temperature':(25,35), 'humidity':(92,100),'ph':(6.0,7.0), 'rainfall':(100,200)},
    'coconut':    {'N':(0,10),   'P':(10,40),  'K':(30,60),  'temperature':(27,37), 'humidity':(90,100),'ph':(5.0,8.0), 'rainfall':(100,200)},
    'cotton':     {'N':(100,140),'P':(35,55),  'K':(15,35),  'temperature':(21,35), 'humidity':(55,80), 'ph':(6.0,8.0), 'rainfall':(50,100)},
    'jute':       {'N':(60,100), 'P':(35,60),  'K':(30,50),  'temperature':(24,37), 'humidity':(70,90), 'ph':(6.0,8.0), 'rainfall':(150,250)},
    'coffee':     {'N':(80,120), 'P':(20,40),  'K':(15,30),  'temperature':(15,28), 'humidity':(85,95), 'ph':(6.0,7.0), 'rainfall':(150,250)},
}

FEATURE_LABELS = {
    'N': 'Nitrogen', 'P': 'Phosphorus', 'K': 'Potassium',
    'temperature': 'Temperature', 'humidity': 'Humidity',
    'ph': 'pH Level', 'rainfall': 'Rainfall'
}


def get_suitability_label(score: float) -> Tuple[str, str]:
    """Map suitability score to label and color."""
    if score >= 90:   return "Excellent", "#00C853"
    elif score >= 80: return "Very Good", "#64DD17"
    elif score >= 70: return "Good",      "#FFD600"
    elif score >= 60: return "Moderate",  "#FF6D00"
    else:             return "Poor",      "#D50000"


def predict_crop(model, scaler, label_encoder,
                 N, P, K, temperature, humidity, ph, rainfall) -> Dict:
    """
    Run full prediction pipeline.

    Returns
    -------
    dict with keys:
        predicted_crop, confidence, top3, suitability_score,
        suitability_label, suitability_color, reasoning, input_values
    """
    input_array = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_array)

    # Class probabilities
    proba = model.predict_proba(input_scaled)[0]
    top3_idx = np.argsort(proba)[::-1][:3]

    predicted_idx  = top3_idx[0]
    predicted_crop = label_encoder.inverse_transform([predicted_idx])[0]
    confidence     = round(float(proba[predicted_idx]) * 100, 2)

    # Top 3 crops
    top3 = []
    for idx in top3_idx:
        crop_name = label_encoder.inverse_transform([idx])[0]
        crop_conf = round(float(proba[idx]) * 100, 2)
        top3.append({"crop": crop_name, "confidence": crop_conf})

    # Suitability score = confidence (prob × 100)
    suitability_score = confidence
    suitability_label, suitability_color = get_suitability_label(suitability_score)

    # Build reasoning
    input_values = {
        'N': N, 'P': P, 'K': K,
        'temperature': temperature, 'humidity': humidity,
        'ph': ph, 'rainfall': rainfall
    }
    reasoning = generate_reasoning(predicted_crop, input_values)

    return {
        "predicted_crop"   : predicted_crop,
        "confidence"       : confidence,
        "top3"             : top3,
        "suitability_score": suitability_score,
        "suitability_label": suitability_label,
        "suitability_color": suitability_color,
        "reasoning"        : reasoning,
        "input_values"     : input_values,
        "input_scaled"     : input_scaled,
    }


def generate_reasoning(crop_name: str, input_values: Dict) -> Dict:
    """
    Generate human-readable reasons why the crop was recommended.

    Returns dict with 'positive_reasons', 'caution_reasons', 'feature_scores'
    """
    ideal = CROP_IDEAL_RANGES.get(crop_name.lower(), {})

    positive_reasons  = []
    caution_reasons   = []
    feature_scores    = {}

    feature_weights = {
        'rainfall': 0.22, 'humidity': 0.18, 'temperature': 0.17,
        'ph': 0.15, 'N': 0.12, 'K': 0.09, 'P': 0.07
    }

    for feat, val in input_values.items():
        label = FEATURE_LABELS.get(feat, feat)
        if feat not in ideal:
            feature_scores[label] = 50
            continue

        lo, hi  = ideal[feat]
        mid     = (lo + hi) / 2
        rng     = (hi - lo) / 2 if (hi - lo) > 0 else 1

        # Score 0-100 based on proximity to ideal range
        if lo <= val <= hi:
            score = 100 - (abs(val - mid) / rng) * 20
            positive_reasons.append(f"✓ {label}: {val:.1f} (ideal range {lo}–{hi})")
        else:
            deviation = min(abs(val - lo), abs(val - hi))
            score     = max(0, 60 - (deviation / rng) * 40)
            caution_reasons.append(f"⚠ {label}: {val:.1f} (ideal {lo}–{hi})")

        feature_scores[label] = round(score, 1)

    # Feature contributions (weighted)
    contributions = {}
    total_weight  = sum(feature_weights.values())
    for feat, weight in feature_weights.items():
        label = FEATURE_LABELS.get(feat, feat)
        score = feature_scores.get(label, 50)
        contributions[label] = round((score / 100) * weight / total_weight * 100, 1)

    return {
        "positive_reasons": positive_reasons[:4],
        "caution_reasons" : caution_reasons[:2],
        "feature_scores"  : feature_scores,
        "contributions"   : contributions,
    }


def validate_inputs(N, P, K, temperature, humidity, ph, rainfall) -> List[str]:
    """Validate user inputs and return list of error messages."""
    errors = []
    if not (0 <= N <= 200):    errors.append("Nitrogen (N) must be between 0–200 mg/kg")
    if not (0 <= P <= 200):    errors.append("Phosphorus (P) must be between 0–200 mg/kg")
    if not (0 <= K <= 200):    errors.append("Potassium (K) must be between 0–200 mg/kg")
    if not (0 <= temperature <= 50):  errors.append("Temperature must be between 0–50 °C")
    if not (0 <= humidity <= 100):    errors.append("Humidity must be between 0–100 %")
    if not (0 <= ph <= 14):           errors.append("pH must be between 0–14")
    if not (0 <= rainfall <= 500):    errors.append("Rainfall must be between 0–500 mm")
    return errors
