"""
Layer 3 orchestration: model training and evaluation on MetroPT-3 windows.

Lineup:
- Decision Tree: fixed params, one fit
- Random Forest: fixed params, one fit
- ExtraTrees: RandomizedSearchCV on 50K subsample, final fit on full train
- XGBoost: RandomizedSearchCV on 50K subsample, final fit on full train

No SVM and no SMOTE in this training layer.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

# Allow running as script: `python src/train.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import run_preprocessing
from utils.constants import HORIZON_OPTIONS, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, SUBSAMPLE_N
from utils.io_utils import load_splits_from_npy, save_json, save_model_artifacts
from utils.training_utils import (
    build_decision_tree_baseline,
    build_extra_trees_search,
    build_horizon_extra_trees,
    build_random_forest_fixed,
    build_xgboost_search,
    compute_scale_pos_weight,
    evaluate_classifier,
    fit_xgboost_search_with_fallback,
    get_stratified_subsample,
)

warnings.filterwarnings("ignore")


def load_splits():
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits_from_npy(PROCESSED_DIR)
    print(f"Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")
    print(f"Train class dist: {np.bincount(y_train)}")
    print(f"Val class dist:   {np.bincount(y_val)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def _print_metrics(label: str, metrics: dict) -> None:
    print(
        f"  [{label}] Acc={metrics['accuracy']} | P={metrics['precision']} | "
        f"R={metrics['recall']} | F1={metrics['f1']} | AUC={metrics['roc_auc']}"
    )


def train_decision_tree(X_train, y_train, X_val, y_val, X_test, y_test):
    """Fixed params baseline Decision Tree."""
    print(f"\n{'=' * 55}\nDECISION TREE (fixed params)\n{'=' * 55}")
    model = build_decision_tree_baseline(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    print("Fitted.")
    val_metrics = evaluate_classifier(model, X_val, y_val, "val")
    test_metrics = evaluate_classifier(model, X_test, y_test, "test")
    _print_metrics("val", val_metrics)
    _print_metrics("test", test_metrics)
    return model, val_metrics, test_metrics, {}


def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    """Fixed params Random Forest for thesis narrative."""
    print(f"\n{'=' * 55}\nRANDOM FOREST (fixed params)\n{'=' * 55}")
    model = build_random_forest_fixed(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    print("Fitted.")
    val_metrics = evaluate_classifier(model, X_val, y_val, "val")
    test_metrics = evaluate_classifier(model, X_test, y_test, "test")
    _print_metrics("val", val_metrics)
    _print_metrics("test", test_metrics)
    return model, val_metrics, test_metrics, {}


def train_extra_trees(X_train, y_train, X_val, y_val, X_test, y_test):
    """Tune on subsample then refit best params on full train set."""
    print(f"\n{'=' * 55}\nEXTRA TREES (search on {SUBSAMPLE_N:,} rows)\n{'=' * 55}")
    X_s, y_s = get_stratified_subsample(X_train, y_train, n=SUBSAMPLE_N, random_state=RANDOM_STATE)
    print(f"  Search subsample: {len(y_s):,} rows | class dist: {np.bincount(y_s)}")

    estimator, param_dist, n_iter = build_extra_trees_search(random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    search.fit(X_s, y_s)
    print(f"Best params: {search.best_params_}")
    print(f"Search CV F1: {search.best_score_:.4f}")

    best_params = search.best_params_
    print(f"Final fit on full {len(y_train):,} rows...")
    model = estimator.set_params(**best_params)
    model.fit(X_train, y_train)
    val_metrics = evaluate_classifier(model, X_val, y_val, "val")
    test_metrics = evaluate_classifier(model, X_test, y_test, "test")
    _print_metrics("val", val_metrics)
    _print_metrics("test", test_metrics)
    return model, val_metrics, test_metrics, best_params


def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """Tune on subsample then refit best params on full train set."""
    print(f"\n{'=' * 55}\nXGBOOST (search on {SUBSAMPLE_N:,} rows)\n{'=' * 55}")
    scale_pos_weight = compute_scale_pos_weight(y_train)
    print(f"  scale_pos_weight: {scale_pos_weight}")

    X_s, y_s = get_stratified_subsample(X_train, y_train, n=SUBSAMPLE_N, random_state=RANDOM_STATE)
    print(f"  Search subsample: {len(y_s):,} rows | class dist: {np.bincount(y_s)}")

    estimator, param_dist, n_iter = build_xgboost_search(
        scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="f1",
        n_jobs=1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    search = fit_xgboost_search_with_fallback(
        search,
        X_s,
        y_s,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
    )
    print(f"Best params: {search.best_params_}")
    print(f"Search CV F1: {search.best_score_:.4f}")

    best_params = search.best_params_
    print(f"Final fit on full {len(y_train):,} rows...")
    model = search.best_estimator_.__class__(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        device=getattr(search.best_estimator_, "device", "cpu"),
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    val_metrics = evaluate_classifier(model, X_val, y_val, "val")
    test_metrics = evaluate_classifier(model, X_test, y_test, "test")
    _print_metrics("val", val_metrics)
    _print_metrics("test", test_metrics)
    return model, val_metrics, test_metrics, best_params


def train_all(X_train, y_train, X_val, y_val, X_test, y_test):
    os.makedirs(MODELS_DIR, exist_ok=True)
    trainers = {
        "decision_tree": train_decision_tree,
        "random_forest": train_random_forest,
        "extra_trees": train_extra_trees,
        "xgboost": train_xgboost,
    }

    results = {}
    best_name = None
    best_f1 = -1.0
    best_model = None
    for name, trainer in trainers.items():
        model, val_metrics, test_metrics, best_params = trainer(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        results[name] = {
            "best_params": {
                key: (int(value) if hasattr(value, "item") else value)
                for key, value in best_params.items()
            },
            "validation": val_metrics,
            "test": test_metrics,
        }
        joblib.dump(model, f"{MODELS_DIR}/{name}.pkl")
        print(f"Saved: {MODELS_DIR}/{name}.pkl")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_name = name
            best_model = model

    print(f"\n{'=' * 55}")
    print(f"BEST: {best_name.upper()} — Val F1: {best_f1}")
    print(f"{'=' * 55}")

    save_model_artifacts(MODELS_DIR, best_name, best_model, best_f1)
    save_json(f"{MODELS_DIR}/evaluation_results.json", results)
    print_summary(results, best_name)
    return results, best_name


def run_horizon_comparison():
    """Compare 1hr/6hr/24hr horizons using fixed ExtraTrees."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    horizon_results = {}
    for horizon in HORIZON_OPTIONS:
        print(f"\n{'#' * 60}\nHORIZON: {horizon}\n{'#' * 60}")
        splits = run_preprocessing(horizon_key=horizon)
        X_train, y_train = splits["X_train"], splits["y_train"]
        X_val, y_val = splits["X_val"], splits["y_val"]
        X_test, y_test = splits["X_test"], splits["y_test"]

        model = build_horizon_extra_trees(RANDOM_STATE)
        model.fit(X_train, y_train)
        val_metrics = evaluate_classifier(model, X_val, y_val, f"val_{horizon}")
        test_metrics = evaluate_classifier(model, X_test, y_test, f"test_{horizon}")
        horizon_results[horizon] = {
            "n_fault_train": int((y_train == 1).sum()),
            "n_normal_train": int((y_train == 0).sum()),
            "val_f1": val_metrics["f1"],
            "val_recall": val_metrics["recall"],
            "val_precision": val_metrics["precision"],
            "val_auc": val_metrics["roc_auc"],
            "test_f1": test_metrics["f1"],
            "test_recall": test_metrics["recall"],
        }

    print(f"\n{'=' * 65}\nHORIZON COMPARISON (ExtraTrees)\n{'=' * 65}")
    print(f"{'Horizon':<10} {'Fault rows':>10} {'Val F1':>8} {'Val Recall':>11} {'Val AUC':>9} {'Test F1':>8}")
    print("-" * 65)
    for horizon, metrics in horizon_results.items():
        print(
            f"{horizon:<10} {metrics['n_fault_train']:>10,} {metrics['val_f1']:>8} "
            f"{metrics['val_recall']:>11} {metrics['val_auc']:>9} {metrics['test_f1']:>8}"
        )

    save_json(f"{MODELS_DIR}/horizon_comparison.json", horizon_results)
    return horizon_results


def print_summary(results: dict, best_name: str):
    print(f"\n{'=' * 72}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'=' * 72}")
    print(
        f"{'Model':<20} {'Val P':>7} {'Val R':>7} {'Val F1':>7} "
        f"{'Val AUC':>8} {'Test F1':>8} {'Test R':>8}"
    )
    print("-" * 72)
    for name, res in results.items():
        marker = " <-- BEST" if name == best_name else ""
        val = res["validation"]
        test = res["test"]
        print(
            f"{name:<20} {val['precision']:>7} {val['recall']:>7} "
            f"{val['f1']:>7} {val['roc_auc']:>8} {test['f1']:>8} {test['recall']:>8}{marker}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "horizon_compare"], default="train")
    args = parser.parse_args()

    if args.mode == "horizon_compare":
        run_horizon_comparison()
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
        train_all(X_train, y_train, X_val, y_val, X_test, y_test)
