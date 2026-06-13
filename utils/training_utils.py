"""Reusable training and evaluation helpers."""

from __future__ import annotations

import numpy as np
from scipy.stats import randint, uniform
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def compute_scale_pos_weight(y: np.ndarray) -> int:
    """Compute XGBoost scale_pos_weight from observed class distribution."""
    n_normal = int((y == 0).sum())
    n_fault = int((y == 1).sum())
    return max(n_normal // max(n_fault, 1), 1)


def get_stratified_subsample(X: np.ndarray, y: np.ndarray, n: int, random_state: int):
    """Return stratified subsample capped at n rows for search."""
    if len(y) <= n:
        return X, y
    X_s, _, y_s, _ = train_test_split(
        X,
        y,
        train_size=n,
        stratify=y,
        random_state=random_state,
    )
    return X_s, y_s


def build_decision_tree_baseline(random_state: int) -> DecisionTreeClassifier:
    """Return fixed Decision Tree baseline model."""
    return DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=8,
        criterion="entropy",
        class_weight="balanced",
        random_state=random_state,
    )


def build_random_forest_fixed(random_state: int) -> RandomForestClassifier:
    """Return fixed Random Forest model for thesis narrative."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        max_samples=0.7,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def build_extra_trees_search(random_state: int) -> tuple[ExtraTreesClassifier, dict, int]:
    """Return estimator, param space, and n_iter for ExtraTrees search."""
    estimator = ExtraTreesClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    param_dist = {
        "n_estimators": randint(100, 300),
        "max_depth": [10, 15, 20, None],
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", 0.5],
    }
    return estimator, param_dist, 10


def build_xgboost_search(scale_pos_weight: int, random_state: int) -> tuple[XGBClassifier, dict, int]:
    """Return estimator, param space, and n_iter for XGBoost search."""
    estimator = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        device="cuda",
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
    )
    param_dist = {
        "n_estimators": randint(100, 300),
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.19),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 10),
        "reg_alpha": uniform(0.0, 0.5),
        "reg_lambda": uniform(0.5, 1.5),
    }
    return estimator, param_dist, 15


def fit_xgboost_search_with_fallback(
    search: RandomizedSearchCV, X_s: np.ndarray, y_s: np.ndarray, scale_pos_weight: int, random_state: int
) -> RandomizedSearchCV:
    """Fit XGBoost search with CUDA first, then CPU fallback if needed."""
    try:
        search.fit(X_s, y_s)
        return search
    except Exception as exc:
        print(f"  XGBoost GPU unavailable ({exc}); retrying on CPU...")
        search.estimator = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            device="cpu",
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0,
        )
        search.fit(X_s, y_s)
        return search


def evaluate_classifier(model, X, y, split_name: str) -> dict:
    """Compute core binary classification metrics for a split."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "split": split_name,
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y, y_pred, average="binary"), 4),
        "roc_auc": round(roc_auc_score(y, y_prob), 4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": classification_report(y, y_pred, output_dict=True, zero_division=0),
    }


def build_horizon_extra_trees(random_state: int) -> ExtraTreesClassifier:
    """Return fixed ExtraTrees model used in horizon comparison experiment."""
    return ExtraTreesClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

