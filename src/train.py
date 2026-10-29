"""Train, tune, calibrate and compare PitchIQ xG models.

The evaluation protocol is grouped by ``match_id`` to prevent shot-level
leakage between train and test. Penalties are excluded from the main model.

Run ``python -m src.train``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import FEATURES_PARQUET, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, SEED, TEST_SPLIT_SEED
from .features import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_preprocessor,
    get_X_y,
)

TEST_SIZE = 0.2
N_SPLITS = 5
N_ITER_XGB = 20
N_ITER_RF = 10
EPS = 1e-6


# ---------------------------------------------------------------------------
# Metrics and calibration helpers
# ---------------------------------------------------------------------------


def _clip(probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
    """Clip probabilities away from 0 and 1 for log loss."""
    return np.clip(np.asarray(probabilities, dtype=float), EPS, 1.0 - EPS)


def classification_metrics(
    y_true: Sequence[int] | pd.Series, probabilities: Sequence[float] | np.ndarray
) -> dict[str, float]:
    """Return ROC-AUC, log loss and Brier score."""
    y_array = np.asarray(y_true, dtype=int)
    p_array = _clip(probabilities)
    return {
        "roc_auc": float(roc_auc_score(y_array, p_array)),
        "log_loss": float(log_loss(y_array, p_array)),
        "brier": float(brier_score_loss(y_array, p_array)),
    }


def calibration_summary(
    y_true: Sequence[int] | pd.Series,
    probabilities: Sequence[float] | np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Return expected calibration error and calibration slope/intercept.

    ECE uses quantile bins so each bin has a similar number of shots. The slope
    comes from regressing the outcome on the logit of the predicted xG; a value
    of 1 means perfect calibration.
    """
    y_array = np.asarray(y_true, dtype=int)
    p_array = _clip(probabilities)
    frame = pd.DataFrame({"y": y_array, "p": p_array})
    frame["bin"] = pd.qcut(frame["p"], q=n_bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(
        n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean")
    )
    ece = float(
        (
            grouped["n"]
            / len(frame)
            * (grouped["observed"] - grouped["predicted"]).abs()
        ).sum()
    )

    logits = np.log(p_array / (1.0 - p_array))
    try:
        slope_model = LogisticRegression(
            C=np.inf, solver="lbfgs", max_iter=1000
        ).fit(logits.reshape(-1, 1), y_array)
        slope = float(slope_model.coef_[0, 0])
        intercept = float(slope_model.intercept_[0])
    except Exception:
        slope = float("nan")
        intercept = float("nan")

    return {"ece": ece, "calibration_slope": slope, "calibration_intercept": intercept}


@dataclass
class ProbabilityCalibrator:
    """Small wrapper so a fitted calibrator can be saved with the model."""

    method: str
    estimator: Any

    def predict(self, probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
        p_array = _clip(probabilities)
        if self.method == "isotonic":
            return np.clip(self.estimator.predict(p_array), 0.0, 1.0)
        logits = np.log(p_array / (1.0 - p_array)).reshape(-1, 1)
        return self.estimator.predict_proba(logits)[:, 1]


def fit_calibrator(
    method: str,
    probabilities: Sequence[float] | np.ndarray,
    y_true: Sequence[int] | pd.Series,
) -> ProbabilityCalibrator:
    """Fit isotonic or sigmoid calibration on out-of-fold predictions."""
    p_array = _clip(probabilities)
    y_array = np.asarray(y_true, dtype=int)
    if method == "isotonic":
        estimator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        estimator.fit(p_array, y_array)
        return ProbabilityCalibrator(method="isotonic", estimator=estimator)

    logits = np.log(p_array / (1.0 - p_array)).reshape(-1, 1)
    estimator = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    estimator.fit(logits, y_array)
    return ProbabilityCalibrator(method="sigmoid", estimator=estimator)


def out_of_fold_predictions(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Generate out-of-fold probabilities on the training set."""
    oof = np.full(len(X), np.nan)
    for train_index, valid_index in folds:
        fitted = clone(model).fit(X.iloc[train_index], y.iloc[train_index])
        oof[valid_index] = fitted.predict_proba(X.iloc[valid_index])[:, 1]
    return oof


def _calibration_cv_brier(
    method: str,
    oof: np.ndarray,
    y: pd.Series,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Cross-validated Brier score for a calibration method fitted on OOF."""
    scores: list[float] = []
    for train_index, valid_index in folds:
        calibrator = fit_calibrator(
            method, oof[train_index], y.iloc[train_index]
        )
        scores.append(
            brier_score_loss(
                y.iloc[valid_index], calibrator.predict(oof[valid_index])
            )
        )
    return float(np.mean(scores))


def calibrate_if_needed(
    model: Pipeline,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Pipeline, ProbabilityCalibrator | None, pd.DataFrame]:
    """Choose calibration using training-only OOF predictions.

    The test set is used only to report final metrics. It must not decide
    whether calibration is applied or which calibration method is selected.
    """
    raw_probabilities = model.predict_proba(X_test)[:, 1]
    raw_row = {
        "model": model_name,
        "variant": "raw",
        **classification_metrics(y_test, raw_probabilities),
        **calibration_summary(y_test, raw_probabilities),
    }

    # Calibration policy is based only on grouped out-of-fold predictions from
    # the training partition. This keeps the final holdout out of model choices.
    oof = out_of_fold_predictions(model, X_train, y_train, folds)
    if not np.isfinite(oof).all():
        raise ValueError("OOF prediction vector contains missing or non-finite values")
    oof_calibration = calibration_summary(y_train, oof)
    well_calibrated = (
        oof_calibration["ece"] <= 0.02
        and 0.8 <= oof_calibration["calibration_slope"] <= 1.2
    )
    if well_calibrated:
        return model, None, pd.DataFrame([raw_row])
    raw_cv_brier = float(
        np.mean(
            [
                brier_score_loss(y_train.iloc[valid], oof[valid])
                for _, valid in folds
            ]
        )
    )
    rows = [raw_row | {"oof_cv_brier": raw_cv_brier}]
    best_method = "raw"
    best_score = raw_cv_brier
    for method in ("isotonic", "sigmoid"):
        score = _calibration_cv_brier(method, oof, y_train, folds)
        fitted_calibrator = fit_calibrator(method, oof, y_train)
        calibrated_probabilities = fitted_calibrator.predict(raw_probabilities)
        rows.append(
            {
                "model": model_name,
                "variant": method,
                **classification_metrics(y_test, calibrated_probabilities),
                **calibration_summary(y_test, calibrated_probabilities),
                "oof_cv_brier": score,
            }
        )
        if score < best_score - 0.0005:
            best_score = score
            best_method = method

    comparison = pd.DataFrame(rows)
    if best_method == "raw":
        return model, None, comparison

    calibrator = fit_calibrator(best_method, oof, y_train)
    return model, calibrator, comparison


# ---------------------------------------------------------------------------
# Cross-validation and ablation
# ---------------------------------------------------------------------------


def cross_validate_metrics(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    """Mean and standard deviation of metrics across GroupKFold folds."""
    records: list[dict[str, float]] = []
    for train_index, valid_index in folds:
        fitted = clone(model).fit(X.iloc[train_index], y.iloc[train_index])
        probabilities = fitted.predict_proba(X.iloc[valid_index])[:, 1]
        records.append(
            classification_metrics(y.iloc[valid_index], probabilities)
        )
    frame = pd.DataFrame(records)
    output: dict[str, float] = {}
    for metric in ("roc_auc", "log_loss", "brier"):
        output[f"cv_{metric}_mean"] = float(frame[metric].mean())
        output[f"cv_{metric}_std"] = float(frame[metric].std(ddof=1))
    return output


ABLATION_STAGES: tuple[tuple[str, list[str], list[str], list[str]], ...] = (
    ("1. Distance + angle", ["distance", "angle_deg"], [], []),
    (
        "2. + location detail",
        [
            "distance",
            "angle_deg",
            "log_distance",
            "distance_to_goal_line",
            "abs_y_from_center",
            "minute",
        ],
        [],
        [],
    ),
    (
        "3. + shot context",
        [
            "distance",
            "angle_deg",
            "log_distance",
            "distance_to_goal_line",
            "abs_y_from_center",
            "minute",
        ],
        BOOLEAN_FEATURES,
        CATEGORICAL_FEATURES,
    ),
    (
        "4. + freeze frame",
        NUMERIC_FEATURES,
        BOOLEAN_FEATURES,
        CATEGORICAL_FEATURES,
    ),
)


def make_logreg(
    numeric: Sequence[str], boolean: Sequence[str], categorical: Sequence[str]
) -> Pipeline:
    """Build a logistic-regression pipeline for a feature subset."""
    return Pipeline(
        steps=[
            (
                "prep",
                build_preprocessor(
                    numeric_features=numeric,
                    boolean_features=boolean,
                    categorical_features=categorical,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED),
            ),
        ]
    )


def run_logreg_ablation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Pipeline, pd.DataFrame]:
    """Fit staged logistic regressions and return the final pipeline + table."""
    rows: list[dict[str, Any]] = []
    final_model: Pipeline | None = None
    for label, numeric, boolean, categorical in ABLATION_STAGES:
        model = make_logreg(numeric, boolean, categorical)
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "stage": label,
                "n_features": len(numeric) + len(boolean) + len(categorical),
                **classification_metrics(y_test, probabilities),
            }
        )
        final_model = model
    table = pd.DataFrame(rows)
    table["delta_auc"] = table["roc_auc"].diff()
    assert final_model is not None
    return final_model, table


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> RandomizedSearchCV:
    """Light randomized search for XGBoost, selected on GroupKFold log loss."""
    pipeline = Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            (
                "clf",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_estimators=400,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    min_child_weight=1,
                    reg_lambda=1.0,
                    n_jobs=1,
                    random_state=SEED,
                ),
            ),
        ]
    )
    parameter_distributions = {
        "clf__n_estimators": [200, 300, 400, 500, 700],
        "clf__max_depth": [2, 3, 4, 5],
        "clf__learning_rate": loguniform(0.01, 0.12),
        "clf__subsample": uniform(0.7, 0.3),
        "clf__colsample_bytree": uniform(0.6, 0.4),
        "clf__min_child_weight": loguniform(1, 20),
        "clf__reg_lambda": loguniform(0.1, 10),
        "clf__gamma": uniform(0, 3),
    }
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_distributions,
        n_iter=N_ITER_XGB,
        scoring="neg_log_loss",
        cv=folds,
        refit=True,
        random_state=SEED,
        n_jobs=4,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> RandomizedSearchCV:
    """Light randomized search for a Random Forest benchmark."""
    pipeline = Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    n_jobs=1,
                    random_state=SEED,
                ),
            ),
        ]
    )
    parameter_distributions = {
        "clf__n_estimators": [300, 500, 700],
        "clf__max_depth": [6, 10, 14, None],
        "clf__min_samples_leaf": [1, 2, 5, 10, 20],
        "clf__max_features": ["sqrt", 0.5, 0.8],
    }
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_distributions,
        n_iter=N_ITER_RF,
        scoring="neg_log_loss",
        cv=folds,
        refit=True,
        random_state=SEED,
        n_jobs=4,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search


# ---------------------------------------------------------------------------
# Evaluation and reporting
# ---------------------------------------------------------------------------


def benchmark_cv_metrics(
    probabilities: np.ndarray,
    y: pd.Series,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    """CV metrics for a fixed prediction vector (StatsBomb xG)."""
    records = [
        classification_metrics(y.iloc[valid], probabilities[valid])
        for _, valid in folds
    ]
    frame = pd.DataFrame(records)
    output: dict[str, float] = {}
    for metric in ("roc_auc", "log_loss", "brier"):
        output[f"cv_{metric}_mean"] = float(frame[metric].mean())
        output[f"cv_{metric}_std"] = float(frame[metric].std(ddof=1))
    return output




def choose_deployment_model(
    available_models: Sequence[str],
    preferred_model: str = "Logistic Regression",
) -> str:
    """Return the pre-declared interpretable deployment baseline.

    XGBoost and Random Forest are tuned with cross-validation. Reusing those
    same folds to choose the shipped model would make their reported CV scores
    optimistically biased unless nested CV is used. Until nested CV is added,
    model family selection is therefore fixed in advance to Logistic Regression;
    test scores and post-tuning CV scores cannot change this choice.
    """
    if preferred_model not in available_models:
        raise ValueError(
            f"Pre-declared deployment model {preferred_model!r} is unavailable"
        )
    return preferred_model

def result_row(
    name: str,
    test_metrics: dict[str, float],
    cv_metrics: dict[str, float],
    calibration: dict[str, float],
) -> dict[str, Any]:
    """Flatten one model's metrics into a table row."""
    return {
        "model": name,
        "test_roc_auc": test_metrics["roc_auc"],
        "test_log_loss": test_metrics["log_loss"],
        "test_brier": test_metrics["brier"],
        "test_ece": calibration["ece"],
        "test_calibration_slope": calibration["calibration_slope"],
        **cv_metrics,
    }


def plot_calibration_curves(
    y_test: pd.Series,
    predictions: dict[str, np.ndarray],
    path: str,
) -> None:
    """Plot reliability curves for the final candidates."""
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    ax.plot([0, 1], [0, 1], "--", color="#9fb3c8", label="Perfect calibration")
    colors = ["#1d3557", "#e63946", "#2a9d8f", "#e9c46a", "#6d597a"]
    for (name, probabilities), color in zip(predictions.items(), colors):
        observed, predicted = calibration_curve(
            y_test, probabilities, n_bins=10, strategy="quantile"
        )
        ax.plot(
            predicted,
            observed,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color,
            label=name,
        )
    ax.set_xlabel("Mean predicted xG")
    ax.set_ylabel("Observed conversion")
    ax.set_title("Calibration on the held-out test matches", loc="left")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.5)
    ax.grid(color="#d9e2ec", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_training() -> None:
    """Run the full Phase 2 modelling workflow."""
    started = time.time()
    pipeline_started_at_utc = datetime.now(timezone.utc).isoformat()
    features = pd.read_parquet(FEATURES_PARQUET)
    X, y = get_X_y(features)
    groups = features.loc[X.index, "match_id"]
    xg_statsbomb = features.loc[X.index, "xg_statsbomb"].to_numpy()

    print("=== 1. GROUPED SPLIT ===")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=TEST_SPLIT_SEED
    )
    train_index, test_index = next(splitter.split(X, y, groups))
    train_matches = set(groups.iloc[train_index])
    test_matches = set(groups.iloc[test_index])
    assert not train_matches.intersection(test_matches)

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print(
        f"Train: {len(X_train):,} shots / {len(train_matches)} matches / "
        f"{int(y_train.sum()):,} goals"
    )
    print(
        f"Test : {len(X_test):,} shots / {len(test_matches)} matches / "
        f"{int(y_test.sum()):,} goals"
    )
    print(f"Leakage check: match overlap = {len(train_matches & test_matches)}")

    cv_splitter = GroupKFold(n_splits=N_SPLITS)
    folds = list(cv_splitter.split(X_train, y_train, groups.iloc[train_index]))
    print(f"GroupKFold: {N_SPLITS} folds on training matches")

    print("\n=== 2. LOGISTIC REGRESSION ABLATION ===")
    logreg_model, ablation = run_logreg_ablation(
        X_train, y_train, X_test, y_test
    )
    display_ablation = ablation.copy()
    for column in ("roc_auc", "log_loss", "brier", "delta_auc"):
        display_ablation[column] = display_ablation[column].map(
            lambda value: f"{value:.4f}" if pd.notna(value) else "-"
        )
    print(display_ablation.to_string(index=False))
    ablation.to_csv(REPORTS_DIR / "logreg_ablation.csv", index=False)

    print("\n=== 3. TUNING XGBOOST ===")
    xgb_search = tune_xgboost(X_train, y_train, folds)
    print(f"Best CV log loss: {-xgb_search.best_score_:.4f}")
    print(f"Best params: {xgb_search.best_params_}")

    print("\n=== 4. TUNING RANDOM FOREST ===")
    rf_search = tune_random_forest(X_train, y_train, folds)
    print(f"Best CV log loss: {-rf_search.best_score_:.4f}")
    print(f"Best params: {rf_search.best_params_}")

    raw_models: dict[str, Pipeline] = {
        "Logistic Regression": logreg_model,
        "XGBoost": xgb_search.best_estimator_,
        "Random Forest": rf_search.best_estimator_,
    }

    print("\n=== 5. TEST + GROUPED CV METRICS ===")
    raw_results: dict[str, dict[str, Any]] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    for name, model in raw_models.items():
        probabilities = model.predict_proba(X_test)[:, 1]
        test_metrics = classification_metrics(y_test, probabilities)
        cv_metrics = cross_validate_metrics(model, X_train, y_train, folds)
        calibration = calibration_summary(y_test, probabilities)
        raw_results[name] = {
            "test": test_metrics,
            "cv": cv_metrics,
            "calibration": calibration,
        }
        test_probabilities[name] = probabilities
        print(
            f"{name}: AUC {test_metrics['roc_auc']:.4f} | "
            f"log loss {test_metrics['log_loss']:.4f} | "
            f"Brier {test_metrics['brier']:.4f} | "
            f"ECE {calibration['ece']:.4f}"
        )

    benchmark_test = classification_metrics(y_test, xg_statsbomb[test_index])
    benchmark_cv = benchmark_cv_metrics(
        xg_statsbomb[train_index], y_train, folds
    )
    benchmark_calibration = calibration_summary(
        y_test, xg_statsbomb[test_index]
    )
    test_probabilities["StatsBomb xG"] = xg_statsbomb[test_index]
    print(
        f"StatsBomb xG: AUC {benchmark_test['roc_auc']:.4f} | "
        f"log loss {benchmark_test['log_loss']:.4f} | "
        f"Brier {benchmark_test['brier']:.4f} | "
        f"ECE {benchmark_calibration['ece']:.4f}"
    )

    print("\n=== 6. DEPLOYMENT MODEL POLICY ===")
    logreg_test = raw_results["Logistic Regression"]["test"]
    logreg_cv = raw_results["Logistic Regression"]["cv"]
    # The model family is declared before looking at evaluation metrics. The
    # tuned tree models use these same folds for hyperparameter search, so their
    # ordinary CV scores are useful for exploration but not a fair selector.
    chosen_base_name = choose_deployment_model(list(raw_results))
    best_cv_name = min(
        raw_results,
        key=lambda name: raw_results[name]["cv"]["cv_brier_mean"],
    )
    best_cv = raw_results[best_cv_name]["cv"]
    best_cv_brier = float(best_cv["cv_brier_mean"])
    logreg_cv_brier = float(logreg_cv["cv_brier_mean"])

    # These gaps are descriptive only; neither test nor post-tuning CV metrics
    # influence the pre-declared Logistic Regression deployment choice.
    best_cv_test = raw_results[best_cv_name]["test"]
    test_auc_gap = best_cv_test["roc_auc"] - logreg_test["roc_auc"]
    test_brier_gap = logreg_test["brier"] - best_cv_test["brier"]
    cv_auc_gap = float(best_cv["cv_roc_auc_mean"] - logreg_cv["cv_roc_auc_mean"])
    cv_brier_gap = best_cv_brier - logreg_cv_brier

    print(f"Lowest reported CV Brier (exploratory only): {best_cv_name}")
    print(f"Logistic Regression grouped CV Brier: {logreg_cv_brier:.4f}")
    print(
        "Deployment model is pre-declared Logistic Regression. Tuned tree-model "
        "CV metrics are not used for selection because this workflow does not "
        "yet use nested CV."
    )
    print(
        f"Diagnostic test deltas (not used for selection): "
        f"AUC {test_auc_gap:+.4f}, Brier {test_brier_gap:+.4f}"
    )
    print(f"Selected base model: {chosen_base_name}")

    print("\n=== 7. CALIBRATION CHECK ===")
    chosen_base_model = raw_models[chosen_base_name]
    calibrated_model, calibrator, calibration_table = calibrate_if_needed(
        chosen_base_model,
        chosen_base_name,
        X_train,
        y_train,
        folds,
        X_test,
        y_test,
    )
    print(calibration_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    calibration_table.to_csv(
        REPORTS_DIR / "calibration_comparison.csv", index=False
    )

    chosen_name = chosen_base_name
    chosen_test = raw_results[chosen_base_name]["test"]
    chosen_calibration = raw_results[chosen_base_name]["calibration"]
    chosen_probabilities = chosen_base_model.predict_proba(X_test)[:, 1]
    if calibrator is not None:
        chosen_name = f"{chosen_base_name} + {calibrator.method}"
        chosen_probabilities = calibrator.predict(chosen_probabilities)
        chosen_test = classification_metrics(y_test, chosen_probabilities)
        chosen_calibration = calibration_summary(
            y_test, chosen_probabilities
        )
        print(
            f"Calibration applied: {chosen_name} "
            f"(AUC {chosen_test['roc_auc']:.4f}, "
            f"Brier {chosen_test['brier']:.4f})"
        )
    else:
        print("Raw probabilities are within tolerance; no calibration applied.")

    candidate_rows: list[dict[str, Any]] = []
    for name, result in raw_results.items():
        candidate_rows.append(
            result_row(
                name, result["test"], result["cv"], result["calibration"]
            )
        )
    if calibrator is not None:
        candidate_rows.append(
            result_row(
                chosen_name,
                chosen_test,
                raw_results[chosen_base_name]["cv"],
                chosen_calibration,
            )
        )
    candidate_rows.append(
        result_row(
            "StatsBomb xG (benchmark)",
            benchmark_test,
            benchmark_cv,
            benchmark_calibration,
        )
    )
    comparison = pd.DataFrame(candidate_rows)
    comparison["recommended"] = comparison["model"].eq(chosen_name)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    print("\n=== 8. FINAL COMPARISON ===")
    display_comparison = comparison.copy()
    for column in display_comparison.columns:
        if column not in ("model", "recommended"):
            display_comparison[column] = display_comparison[column].map(
                lambda value: f"{value:.4f}"
            )
    print(display_comparison.to_string(index=False))

    print("\n=== 9. MODEL RECOMMENDATION ===")
    reason = (
        "Logistic Regression is the pre-declared deployment baseline for "
        "transparency and inspectability. Tuned tree models remain exploratory "
        "comparators; their CV scores are not used to select the deployed model "
        "because tuning and evaluation currently share folds (nested CV is not "
        "implemented). Neither outer-test metrics nor candidate CV metrics "
        "influence this model-family choice."
    )
    print(f"Recommended model: {chosen_name}")
    print(reason)
    print(
        f"Test: AUC {chosen_test['roc_auc']:.4f} | "
        f"log loss {chosen_test['log_loss']:.4f} | "
        f"Brier {chosen_test['brier']:.4f} | "
        f"ECE {chosen_calibration['ece']:.4f}"
    )
    chosen_model = calibrated_model
    chosen_calibrator = calibrator

    fig_predictions = {
        name: test_probabilities[name] for name in raw_models
    }
    fig_predictions["StatsBomb xG"] = test_probabilities["StatsBomb xG"]
    if calibrator is not None:
        fig_predictions[chosen_name] = chosen_probabilities
    plot_calibration_curves(
        y_test,
        fig_predictions,
        str(FIGURES_DIR / "fig5_calibration.png"),
    )

    parameters = {
        "Logistic Regression": {"C": 1.0, "solver": "lbfgs"},
        "XGBoost": xgb_search.best_params_,
        "Random Forest": rf_search.best_params_,
    }
    alternative_name = "XGBoost" if chosen_base_name != "XGBoost" else "Random Forest"
    metadata = {
        "model_name": chosen_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_started_at_utc": pipeline_started_at_utc,
        "seed": SEED,
        "split": {
            "test_split_seed": TEST_SPLIT_SEED,
            "test_size": TEST_SIZE,
            "n_splits": N_SPLITS,
            "train_matches": len(train_matches),
            "test_matches": len(test_matches),
            "train_shots": len(X_train),
            "test_shots": len(X_test),
        },
        "penalties_excluded": True,
        "features": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "test_metrics": chosen_test,
        "calibration": chosen_calibration,
        "cv_metrics": raw_results[chosen_base_name]["cv"],
        "calibration_method": chosen_calibrator.method
        if chosen_calibrator is not None
        else "none",
        "hyperparameters": parameters.get(chosen_base_name, {}),
        "audit_protocol_version": "2026-10-29-v1",
        "selection": {
            "criterion": "pre-declared interpretable baseline (Logistic Regression)",
            "selected_by_rule": chosen_base_name,
            "lowest_reported_cv_brier_exploratory_only": best_cv_name,
            "test_metrics_used_for_selection": False,
            "cv_metrics_used_for_selection": False,
            "nested_cv_for_tuned_candidates": False,
            "test_auc_gap_vs_logreg_descriptive_only": test_auc_gap,
            "test_brier_gap_vs_logreg_descriptive_only": test_brier_gap,
            "cv_auc_gap_vs_logreg_descriptive_only": cv_auc_gap,
            "cv_brier_gap_vs_logreg_descriptive_only": cv_brier_gap,
            "reason": reason,
        },
        "alternative": {
            "model": alternative_name,
            "test_metrics": raw_results[alternative_name]["test"],
            "cv_metrics": raw_results[alternative_name]["cv"],
            "calibration": raw_results[alternative_name]["calibration"],
        },
        "statsbomb_benchmark": {
            **benchmark_test,
            **benchmark_calibration,
        },
        "notes": (
            "Grouped split by match_id. Penalties excluded. StatsBomb xG is a "
            "benchmark, never a feature. StatsBomb xG was trained on external "
            "data, so the comparison is indicative, not a like-for-like "
            "training-data comparison."
        ),
    }
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=_json_default),
        encoding="utf-8",
    )

    artifact = {
        "model_name": chosen_name,
        "pipeline": chosen_model,
        "calibrator": chosen_calibrator,
        "feature_columns": FEATURE_COLUMNS,
        "metadata": metadata,
    }
    joblib.dump(artifact, MODELS_DIR / "best_model.joblib", compress=3)
    print(f"\nSaved model -> {MODELS_DIR / 'best_model.joblib'}")
    print(f"Saved metadata -> {MODELS_DIR / 'metadata.json'}")
    print(f"Saved comparison -> {REPORTS_DIR / 'model_comparison.csv'}")
    print(f"Saved calibration -> {REPORTS_DIR / 'calibration_comparison.csv'}")
    print(f"Saved figure -> {FIGURES_DIR / 'fig5_calibration.png'}")
    print(f"Total runtime: {time.time() - started:.1f}s")


def load_artifact(path: str | None = None) -> dict[str, Any]:
    """Load the saved model artifact (used by the Streamlit app later)."""
    model_path = MODELS_DIR / "best_model.joblib" if path is None else path
    return joblib.load(model_path)


def predict_probabilities(
    artifact: dict[str, Any], features: pd.DataFrame
) -> np.ndarray:
    """Apply the saved pipeline and optional calibrator to a feature frame."""
    columns = artifact["feature_columns"]
    probabilities = artifact["pipeline"].predict_proba(features[columns])[:, 1]
    calibrator = artifact.get("calibrator")
    if calibrator is not None:
        probabilities = calibrator.predict(probabilities)
    return probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PitchIQ xG models.")
    parser.parse_args()
    run_training()


if __name__ == "__main__":
    main()
