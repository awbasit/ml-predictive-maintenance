"""
Layer 6-7 orchestration: model training and horizon comparison.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Allow running as script: `python src/train.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import run_preprocessing
from utils.constants import (
    HORIZON_OPTIONS,
    MODELS_DIR,
    NOISE_STD,
    PROCESSED_DIR,
    RANDOM_STATE,
)
from utils.io_utils import load_splits_from_npy, save_json, save_model_artifacts
from utils.training_utils import (
    add_gaussian_noise,
    build_horizon_random_forest,
    build_learning_curve_artifacts,
    evaluate_classifier,
    get_model_configs,
)

warnings.filterwarnings("ignore")


def load_splits():
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits_from_npy(PROCESSED_DIR)
    print(f"Loaded splits — train: {len(y_train):,}, val: {len(y_val):,}, test: {len(y_test):,}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_all_models(X_train, y_train, X_val, y_val, X_test, y_test):
    os.makedirs(MODELS_DIR, exist_ok=True)
    scale_pos_weight = max(int((y_train == 0).sum() / max((y_train == 1).sum(), 1)), 1)
    configs = get_model_configs(RANDOM_STATE, scale_pos_weight=scale_pos_weight)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    X_train_noisy = add_gaussian_noise(X_train, noise_std=NOISE_STD, random_state=RANDOM_STATE)
    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_model = None

    for name, config in configs.items():
        print(f"\n{'=' * 55}\nTraining: {name.upper()}\n{'=' * 55}")
        search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train_noisy, y_train)
        model = search.best_estimator_
        print(f"Best params: {search.best_params_}")
        print(f"CV F1: {search.best_score_:.4f}")

        val_metrics = evaluate_classifier(model, X_val, y_val, "validation")
        test_metrics = evaluate_classifier(model, X_test, y_test, "test")
        print(
            f"  [validation] P={val_metrics['precision']} R={val_metrics['recall']} "
            f"F1={val_metrics['f1']} AUC={val_metrics['roc_auc']}"
        )
        print(
            f"  [test]       P={test_metrics['precision']} R={test_metrics['recall']} "
            f"F1={test_metrics['f1']} AUC={test_metrics['roc_auc']}"
        )

        results[name] = {
            "best_params": search.best_params_,
            "cv_f1": round(search.best_score_, 4),
            "validation": val_metrics,
            "test": test_metrics,
        }
        joblib.dump(model, f"{MODELS_DIR}/{name}.pkl")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_model_name = name
            best_model = model

    save_model_artifacts(MODELS_DIR, best_model_name, best_model, best_f1)

    lc_plot = f"{MODELS_DIR}/learning_curve_{best_model_name}.png"
    lc_json = f"{MODELS_DIR}/learning_curve_{best_model_name}.json"
    lc_summary = build_learning_curve_artifacts(best_model, X_train_noisy, y_train, cv, lc_plot)
    save_json(lc_json, lc_summary)

    results["learning_curve"] = {
        "model": best_model_name,
        "plot_path": lc_plot,
        "json_path": lc_json,
        "summary": lc_summary,
    }
    save_json(f"{MODELS_DIR}/evaluation_results.json", results)
    print(f"Best model: {best_model_name} (Val F1={best_f1:.4f})")
    return results, best_model_name


def run_horizon_comparison():
    """Compare 1hr/6hr/24hr horizons using fixed random forest."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    comparison = {}
    for horizon in HORIZON_OPTIONS:
        print(f"\n{'#' * 60}\nHORIZON: {horizon}\n{'#' * 60}")
        splits = run_preprocessing(horizon_key=horizon)
        X_train, y_train = splits["X_train"], splits["y_train"]
        X_val, y_val = splits["X_val"], splits["y_val"]
        X_test, y_test = splits["X_test"], splits["y_test"]

        rf = build_horizon_random_forest(RANDOM_STATE)
        rf.fit(X_train, y_train)
        val_metrics = evaluate_classifier(rf, X_val, y_val, f"val_{horizon}")
        test_metrics = evaluate_classifier(rf, X_test, y_test, f"test_{horizon}")

        comparison[horizon] = {
            "n_fault_train": int((y_train == 1).sum()),
            "n_normal_train": int((y_train == 0).sum()),
            "val_f1": val_metrics["f1"],
            "val_recall": val_metrics["recall"],
            "val_precision": val_metrics["precision"],
            "val_auc": val_metrics["roc_auc"],
            "test_f1": test_metrics["f1"],
            "test_recall": test_metrics["recall"],
        }

    save_json(f"{MODELS_DIR}/horizon_comparison.json", comparison)
    return comparison


def print_summary(results: dict, best_model_name: str):
    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<20} {'Val P':>7} {'Val R':>7} {'Val F1':>8} {'Val AUC':>8} {'Test F1':>8}")
    print("-" * 70)
    for name, res in results.items():
        if name == "learning_curve":
            continue
        marker = " <-- BEST" if name == best_model_name else ""
        val = res["validation"]
        test = res["test"]
        print(
            f"{name:<20} {val['precision']:>7} {val['recall']:>7} "
            f"{val['f1']:>8} {val['roc_auc']:>8} {test['f1']:>8}{marker}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "horizon_compare"], default="train")
    args = parser.parse_args()

    if args.mode == "horizon_compare":
        run_horizon_comparison()
    else:
        # Default expects preprocessing already run with DEFAULT_HORIZON.
        X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
        results, best_name = train_all_models(X_train, y_train, X_val, y_val, X_test, y_test)
        print_summary(results, best_name)
