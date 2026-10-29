"""Regression tests for pre-declared model selection policy."""

from __future__ import annotations

import pytest

from src.train import choose_deployment_model


def test_deployment_choice_ignores_test_and_post_tuning_cv_metrics() -> None:
    # The model-choice helper accepts only available names, not metrics, making
    # it impossible for test/CV scores to silently influence deployment choice.
    available = ["Logistic Regression", "XGBoost", "Random Forest"]
    assert choose_deployment_model(available) == "Logistic Regression"


def test_deployment_choice_requires_predeclared_model_to_be_available() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        choose_deployment_model(["XGBoost", "Random Forest"])
