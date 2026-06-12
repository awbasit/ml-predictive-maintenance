"""
Layer 3 orchestration: model training, tuning, and evaluation.
Reusable training logic lives under utils/.
"""

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

from utils.constants import MODELS_DIR, PROCESSED_DIR, RANDOM_STATE
from utils.io_utils import load_splits_from_npy, save_json
from utils.training_utils import (
    add_gaussian_noise,
    build_learning_curve_artifacts,
    evaluate_classifier,
    get_model_configs,
)

warnings.filterwarnings("ignore")


def load_splits():
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits_from_npy(PROCESSED_DIR)
    print(f"Loaded splits — train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_all_models(X_train, y_train, X_val, y_val, X_test, y_test):
    os.makedirs(MODELS_DIR, exist_ok=True)
    configs = get_model_configs(RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    X_train_noisy = add_gaussian_noise(X_train, noise_std=0.03, random_state=RANDOM_STATE)

    results = {}
    best_model_name = None
    best_f1 = -1.0

    for name, config in configs.items():
        print(f"\n{'=' * 50}")
        print(f"Training: {name.upper()}")
        print(f"{'=' * 50}")

        search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train_noisy, y_train)
        best = search.best_estimator_

        print(f"Best params: {search.best_params_}")
        print(f"CV F1: {round(search.best_score_, 4)}")

        val_metrics = evaluate_classifier(best, X_val, y_val, "validation")
        test_metrics = evaluate_classifier(best, X_test, y_test, "test")
        print(f"  [validation] Accuracy: {val_metrics['accuracy']} | F1: {val_metrics['f1']} | ROC-AUC: {val_metrics['roc_auc']}")
        print(f"  [test] Accuracy: {test_metrics['accuracy']} | F1: {test_metrics['f1']} | ROC-AUC: {test_metrics['roc_auc']}")

        results[name] = {
            "best_params": search.best_params_,
            "cv_f1": round(search.best_score_, 4),
            "validation": val_metrics,
            "test": test_metrics,
        }

        joblib.dump(best, f"{MODELS_DIR}/{name}.pkl")
        print(f"Saved: {MODELS_DIR}/{name}.pkl")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_model_name = name

    print(f"\n{'=' * 50}")
    print(f"BEST MODEL: {best_model_name.upper()} (Val F1: {best_f1})")
    print(f"{'=' * 50}")

    best_model = joblib.load(f"{MODELS_DIR}/{best_model_name}.pkl")
    joblib.dump(best_model, f"{MODELS_DIR}/best_model.pkl")
    joblib.dump({"name": best_model_name, "val_f1": best_f1}, f"{MODELS_DIR}/best_model_meta.pkl")

    learning_curve_plot_path = f"{MODELS_DIR}/learning_curve_{best_model_name}.png"
    learning_curve_data = build_learning_curve_artifacts(
        estimator=best_model,
        X=X_train_noisy,
        y=y_train,
        cv=cv,
        output_plot_path=learning_curve_plot_path,
    )
    save_json(f"{MODELS_DIR}/learning_curve_{best_model_name}.json", learning_curve_data)
    results["learning_curve"] = {
        "model": best_model_name,
        "plot_path": learning_curve_plot_path,
        "json_path": f"{MODELS_DIR}/learning_curve_{best_model_name}.json",
        "summary": learning_curve_data,
    }
    save_json(f"{MODELS_DIR}/evaluation_results.json", results)

    print(f"\nAll results saved to: {MODELS_DIR}/evaluation_results.json")
    print(f"Learning curve plot saved to: {learning_curve_plot_path}")
    return results, best_model_name


def print_summary(results: dict, best_model_name: str):
    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    header = f"{'Model':<20} {'Val Acc':>8} {'Val F1':>8} {'Val AUC':>9} {'Test F1':>8}"
    print(header)
    print("-" * 60)
    for name, res in results.items():
        marker = " <-- BEST" if name == best_model_name else ""
        print(
            f"{name:<20} "
            f"{res['validation']['accuracy']:>8} "
            f"{res['validation']['f1']:>8} "
            f"{res['validation']['roc_auc']:>9} "
            f"{res['test']['f1']:>8}"
            f"{marker}"
        )


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    results, best_model_name = train_all_models(X_train, y_train, X_val, y_val, X_test, y_test)
    print_summary(results, best_model_name)
