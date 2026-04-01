import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.model import compute_model_metrics, inference, train_model


@pytest.fixture
def simple_training_data():
    """Small deterministic dataset for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_train_model_returns_random_forest(simple_training_data):
    """train_model should return a fitted RandomForestClassifier."""
    X, y = simple_training_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), (
        f"Expected RandomForestClassifier, got {type(model)}"
    )


def test_inference_returns_numpy_array_of_correct_shape(simple_training_data):
    """inference should return a numpy array with one prediction per sample."""
    X, y = simple_training_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray), (
        f"Expected np.ndarray, got {type(preds)}"
    )
    assert preds.shape == (len(X),), (
        f"Expected shape ({len(X)},), got {preds.shape}"
    )


def test_compute_model_metrics_perfect_predictions():
    """compute_model_metrics should return 1.0 for precision, recall, and F1
    when predictions exactly match the labels."""
    y = np.array([0, 1, 0, 1, 1])
    preds = np.array([0, 1, 0, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(1.0), f"Expected precision 1.0, got {precision}"
    assert recall == pytest.approx(1.0), f"Expected recall 1.0, got {recall}"
    assert fbeta == pytest.approx(1.0), f"Expected F1 1.0, got {fbeta}"


def test_compute_model_metrics_known_values():
    """compute_model_metrics should return correct values for a known
    prediction set: 2 true positives, 1 false positive, 1 false negative."""
    y =     np.array([1, 1, 1, 0])
    preds = np.array([1, 1, 0, 1])
    # TP=2, FP=1, FN=1 → precision=2/3, recall=2/3, F1=2/3
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(2 / 3, abs=1e-4)
    assert recall == pytest.approx(2 / 3, abs=1e-4)
    assert fbeta == pytest.approx(2 / 3, abs=1e-4)


def test_inference_predictions_are_binary(simple_training_data):
    """inference predictions should only contain values 0 and 1 for a
    binary classifier."""
    X, y = simple_training_data
    model = train_model(X, y)
    preds = inference(model, X)
    unique_values = set(np.unique(preds))
    assert unique_values.issubset({0, 1}), (
        f"Expected only {{0, 1}} in predictions, got {unique_values}"
    )
