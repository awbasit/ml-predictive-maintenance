"""Reusable training and evaluation helpers."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def get_model_configs(random_state: int) -> dict:
    """Return model registry and hyperparameter grids."""
    return {
        "random_forest": {
            "model": RandomForestClassifier(random_state=random_state, n_jobs=-1),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "class_weight": ["balanced"],
            },
        },
        "decision_tree": {
            "model": DecisionTreeClassifier(random_state=random_state),
            "params": {
                "max_depth": [3, 4, 5, 6],
                "min_samples_split": [8, 12, 20],
                "min_samples_leaf": [3, 5, 8],
                "criterion": ["gini", "entropy"],
                "class_weight": ["balanced"],
            },
        },
        "svm": {
            "model": SVC(random_state=random_state, probability=True),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"],
                "class_weight": ["balanced"],
            },
        },
        "xgboost": {
            "model": XGBClassifier(
                random_state=random_state,
                eval_metric="logloss",
                verbosity=0,
            ),
            "params": {
                "n_estimators": [120, 200],
                "max_depth": [2, 3, 4],
                "learning_rate": [0.03, 0.08],
                "subsample": [0.7, 0.85],
                "colsample_bytree": [0.7, 0.9],
                "min_child_weight": [3, 6, 10],
                "reg_alpha": [0.0, 0.2, 0.6],
                "scale_pos_weight": [4],
            },
        },
    }


def evaluate_classifier(model, X, y, split_name: str) -> dict:
    """Compute core binary classification metrics for a split."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "split": split_name,
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "f1": round(f1_score(y, y_pred, average="binary"), 4),
        "roc_auc": round(roc_auc_score(y, y_prob), 4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": classification_report(y, y_pred, output_dict=True),
    }


def add_gaussian_noise(X: np.ndarray, noise_std: float, random_state: int) -> np.ndarray:
    """Inject zero-mean Gaussian noise to improve robustness."""
    rng = np.random.default_rng(random_state)
    noise = rng.normal(loc=0.0, scale=noise_std, size=X.shape)
    return X + noise


def build_learning_curve_artifacts(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    output_plot_path: str,
    train_sizes: np.ndarray | None = None,
) -> dict:
    """Compute and save learning-curve metrics and plot."""
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 8)

    sizes, train_scores, val_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        train_sizes=train_sizes,
        scoring="f1",
        n_jobs=-1,
        shuffle=True,
        random_state=42,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, train_mean, marker="o", color="#6a8bff", label="Train F1")
    ax.plot(sizes, val_mean, marker="o", color="#2ed8a3", label="Validation F1")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, color="#6a8bff", alpha=0.18)
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, color="#2ed8a3", alpha=0.18)
    ax.set_xlabel("Training samples")
    ax.set_ylabel("F1 score")
    ax.set_title("Learning Curve (Train vs Validation F1)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_plot_path, dpi=160)
    plt.close(fig)

    return {
        "train_sizes": sizes.tolist(),
        "train_f1_mean": np.round(train_mean, 4).tolist(),
        "train_f1_std": np.round(train_std, 4).tolist(),
        "val_f1_mean": np.round(val_mean, 4).tolist(),
        "val_f1_std": np.round(val_std, 4).tolist(),
    }

