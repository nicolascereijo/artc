from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.tree import plot_tree


def tree_plots(
    model: RandomForestClassifier,
    feature_names: list[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    confusion_matrix_cmap: str = "Blues",
    tree_max_depth: int = 5,
    fig_width: int = 50,
    fig_height: int = 25
) -> Tuple[Figure, Figure, Figure, Figure]:
    return (
        confusion_matrix(model, X_test, y_test, cmap=confusion_matrix_cmap),
        roc_curve(model, X_test, y_test),
        metric_importance(model, feature_names, fig_width=fig_width, fig_height=fig_height),
        decision_tree(model, feature_names, max_depth=tree_max_depth,
            fig_width=fig_width, fig_height=fig_height)
    )


def confusion_matrix(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray, *,
    cmap: str = "Blues") -> Figure:
    fig, ax = plt.subplots()

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=cmap, ax=ax)
    ax.set_title("Random forest confusion matrix")

    return fig


def roc_curve(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Figure:
    fig, ax = plt.subplots()

    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("ROC curve of the model")

    return fig


def metric_importance(model: RandomForestClassifier, feature_names: list[str], *,
    fig_width: int = 50, fig_height: int = 25) -> Figure:
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.bar(range(len(importances)), importances[indices], align='center')
    ax.set_xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    ax.set_title('Feature importance')

    return fig


def decision_tree(model: RandomForestClassifier, feature_names: list[str], *,
    max_depth: int = 5, fig_width: int = 50, fig_height: int = 25) -> Figure:
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    plot_tree(model.estimators_[0], filled=True, max_depth=max_depth,
        feature_names=feature_names, class_names=["0", "1"])
    ax.set_title(f"Forest decision tree (depth limited to {max_depth})")

    return fig
