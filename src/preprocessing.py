"""
preprocessing.py
================
Data preprocessing pipeline for the Explainable Crop Suitability Prediction System.
Handles loading data, cleaning, outlier detection, and feature scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET_COL   = 'label'
RANDOM_STATE = 42
TEST_SIZE    = 0.2


def load_data(filepath: str) -> pd.DataFrame:
    """Load the crop recommendation CSV dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a summary of missing values per column."""
    missing = df.isnull().sum()
    pct     = (missing / len(df) * 100).round(2)
    report  = pd.DataFrame({'missing_count': missing, 'missing_pct': pct})
    return report


def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and remove duplicate rows."""
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    return df, n_dupes


def analyze_outliers(df: pd.DataFrame, columns: list = None) -> dict:
    """Detect outliers using IQR method."""
    if columns is None:
        columns = FEATURE_COLS
    outlier_report = {}
    for col in columns:
        if col not in df.columns:
            continue
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_report[col] = int(n_out)
    return outlier_report


def clean_data(df: pd.DataFrame):
    """Run the complete data cleaning pipeline."""
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    str_cols = df.select_dtypes(include='object').columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    df, n_dupes = detect_duplicates(df)
    missing_report   = analyze_missing_values(df)
    outlier_report   = analyze_outliers(df)
    return df, missing_report, n_dupes, outlier_report


def feature_distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for all numeric features."""
    summary = df[FEATURE_COLS].describe().T
    summary['skewness'] = df[FEATURE_COLS].skew()
    summary['kurtosis'] = df[FEATURE_COLS].kurt()
    return summary


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix for feature columns."""
    return df[FEATURE_COLS].corr()


def prepare_features(df: pd.DataFrame):
    """Encode labels, split, and scale features."""
    le = LabelEncoder()
    df = df.copy()
    df[TARGET_COL] = le.fit_transform(df[TARGET_COL])
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    class_names = list(le.classes_)
    return X_train, X_test, y_train, y_test, scaler, le, class_names


def save_preprocessors(scaler, label_encoder, save_dir: str = "models"):
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaler,        os.path.join(save_dir, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(save_dir, "label_encoder.pkl"))


def load_preprocessors(save_dir: str = "models"):
    scaler = joblib.load(os.path.join(save_dir, "scaler.pkl"))
    le     = joblib.load(os.path.join(save_dir, "label_encoder.pkl"))
    return scaler, le
