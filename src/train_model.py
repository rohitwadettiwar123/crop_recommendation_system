"""
train_model.py
==============
Model training, comparison, and selection.
Compares: Random Forest, XGBoost, LightGBM, Decision Tree, KNN, Logistic Regression
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble        import RandomForestClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.linear_model    import LogisticRegression
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier
from sklearn.metrics         import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

RANDOM_STATE = 42


def get_candidate_models() -> dict:
    return {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost":       XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                       use_label_encoder=False, eval_metric='mlogloss',
                                       random_state=RANDOM_STATE, verbosity=0),
        "LightGBM":      LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, verbose=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        "KNN":           KNeighborsClassifier(n_neighbors=5, weights='distance'),
        "Logistic Regression": LogisticRegression(
          max_iter=1000,
          random_state=RANDOM_STATE
),
    }


def evaluate_model(model, X_train, X_test, y_train, y_test, cv_folds=5) -> dict:
    model.fit(X_train, y_train)
    y_pred    = model.predict(X_test)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds,
                                scoring='f1_weighted', n_jobs=-1)
    return {
        "accuracy" : round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
        "recall"   : round(float(recall_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
        "f1"       : round(float(f1_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
        "cv_mean"  : round(float(cv_scores.mean()), 4),
        "cv_std"   : round(float(cv_scores.std()), 4),
    }


def compare_models(X_train, X_test, y_train, y_test, cv_folds=5):
    candidates     = get_candidate_models()
    results        = {}
    trained_models = {}
    for name, model in candidates.items():
        try:
            metrics              = evaluate_model(model, X_train, X_test, y_train, y_test, cv_folds)
            results[name]        = metrics
            trained_models[name] = model
        except Exception as e:
            print(f"[WARNING] {name} failed: {e}")

    results_df = (pd.DataFrame(results).T.reset_index()
                  .rename(columns={'index':'model'})
                  .sort_values('f1', ascending=False)
                  .reset_index(drop=True))
    return results_df, trained_models


def select_best_model(results_df, trained_models):
    best_row     = results_df.iloc[0]
    best_name    = best_row['model']
    best_model   = trained_models[best_name]
    best_metrics = best_row.drop('model').to_dict()
    return best_name, best_model, best_metrics


def save_model(model, model_name, metrics, class_names, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(save_dir, "crop_model.pkl"))
    metadata = {
        "model_name"  : model_name,
        "metrics"     : metrics,
        "class_names" : class_names,
        "feature_cols": ['N','P','K','temperature','humidity','ph','rainfall'],
    }
    with open(os.path.join(save_dir, "model_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)


def load_model(save_dir="models"):
    model = joblib.load(os.path.join(save_dir, "crop_model.pkl"))
    with open(os.path.join(save_dir, "model_metadata.json"), 'r') as f:
        metadata = json.load(f)
    return model, metadata


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.preprocessing import load_data, clean_data, prepare_features, save_preprocessors

    df, _, _, _ = clean_data(load_data(os.path.join("data","Crop_recommendation.csv")))
    X_train, X_test, y_train, y_test, scaler, le, class_names = prepare_features(df)
    save_preprocessors(scaler, le)
    results_df, trained_models = compare_models(X_train, X_test, y_train, y_test)
    print(results_df[['model','accuracy','f1','cv_mean']].to_string(index=False))
    best_name, best_model, best_metrics = select_best_model(results_df, trained_models)
    save_model(best_model, best_name, best_metrics, class_names)
    print(f"\n[SUCCESS] Best model: {best_name}")
