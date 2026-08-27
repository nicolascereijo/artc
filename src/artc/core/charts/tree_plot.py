import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import rcParams
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.tree import plot_tree


def tree_plots(
    model: RandomForestClassifier,
    feature_names: list[str],
    X_test: npt.NDArray[np.float64],
    y_test: npt.NDArray[np.int64],
    /,
    *,
    confusion_matrix_cmap: str = "Blues",
    tree_max_depth: int = 5,
    fig_width: int = 50,
    fig_height: int = 25,
) -> tuple[Figure, Figure, Figure, Figure]:
    """Builds every diagnostic plot for a fitted random forest in one call.

    Runs 'confusion_matrix', 'roc_curve', 'metric_importance' and
    'decision_tree' against the same model and test set, so a caller who
    wants the full picture doesn't need to invoke each plotting function
    separately.

    Args:
        model: Fitted classifier to plot.
        feature_names: Name of each feature column in 'X_test', in the same
            order the model was trained on.
        X_test: Test set feature matrix.
        y_test: Test set ground truth labels.
        confusion_matrix_cmap: Matplotlib colormap name for the confusion
            matrix cells.
        tree_max_depth: Maximum depth to render when plotting the decision
            tree.
        fig_width: Width, in inches, of the feature importance and decision
            tree figures.
        fig_height: Height, in inches, of the feature importance and
            decision tree figures.

    Returns:
        A tuple '(confusion_matrix_figure, roc_curve_figure,
        metric_importance_figure, decision_tree_figure)'.
    """
    # Set explicitly here (rather than as an import-time side effect) so
    # merely importing this module doesn't silently mutate a global
    # matplotlib setting for callers who never asked for these plots.
    rcParams["font.family"] = "DejaVu Sans"

    return (
        confusion_matrix(model, X_test, y_test, cmap=confusion_matrix_cmap),
        roc_curve(model, X_test, y_test),
        metric_importance(
            model, feature_names, fig_width=fig_width, fig_height=fig_height
        ),
        decision_tree(
            model,
            feature_names,
            max_depth=tree_max_depth,
            fig_width=fig_width,
            fig_height=fig_height,
        ),
    )


def confusion_matrix(
    model: RandomForestClassifier,
    X_test: npt.NDArray[np.float64],
    y_test: npt.NDArray[np.int64],
    /,
    *,
    cmap: str = "Blues",
) -> Figure:
    """Plots the confusion matrix of a fitted classifier against a test set.

    Args:
        model: Fitted classifier to evaluate.
        X_test: Test set feature matrix.
        y_test: Test set ground truth labels.
        cmap: Matplotlib colormap name for the matrix cells.

    Returns:
        The figure holding the confusion matrix plot.
    """
    fig, ax = plt.subplots()  # pyright: ignore[reportAny]

    _ = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, cmap=cmap, ax=ax  # pyright: ignore[reportAny]
    )
    ax.set_title(  # pyright: ignore[reportAny]
        "Random forest confusion matrix"
    )

    return fig


def roc_curve(
    model: RandomForestClassifier,
    X_test: npt.NDArray[np.float64],
    y_test: npt.NDArray[np.int64],
) -> Figure:
    """Plots the ROC curve of a fitted classifier against a test set.

    Args:
        model: Fitted classifier to evaluate.
        X_test: Test set feature matrix.
        y_test: Test set ground truth labels.

    Returns:
        The figure holding the ROC curve plot.
    """
    fig, ax = plt.subplots()  # pyright: ignore[reportAny]

    _ = RocCurveDisplay.from_estimator(
        model, X_test, y_test, ax=ax  # pyright: ignore[reportAny]
    )
    ax.set_title("ROC curve of the model")  # pyright: ignore[reportAny]

    return fig


def metric_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
    /,
    *,
    fig_width: int = 50,
    fig_height: int = 25,
) -> Figure:
    """Plots each feature's importance in a fitted random forest.

    Args:
        model: Fitted classifier to read feature importances from.
        feature_names: Name of each feature column, in the same order the
            model was trained on.
        fig_width: Width, in inches, of the figure.
        fig_height: Height, in inches, of the figure.

    Returns:
        The figure holding the feature importance bar plot, sorted from
        most to least important.
    """
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    fig, ax = plt.subplots(  # pyright: ignore[reportAny]
        figsize=(fig_width, fig_height)
    )

    ax.bar(  # pyright: ignore[reportAny]
        range(len(importances)), importances[indices], align="center"
    )
    ax.set_xticks(  # pyright: ignore[reportAny]
        range(len(importances)), [feature_names[i] for i in indices],
        rotation=90,
    )
    ax.set_title("Feature importance")  # pyright: ignore[reportAny]

    return fig


def decision_tree(
    model: RandomForestClassifier,
    feature_names: list[str],
    /,
    *,
    max_depth: int = 5,
    fig_width: int = 50,
    fig_height: int = 25,
) -> Figure:
    """Plots the first tree of a fitted random forest.

    Args:
        model: Fitted classifier to plot a tree from.
        feature_names: Name of each feature column, in the same order the
            model was trained on.
        max_depth: Maximum depth to render, deeper splits are collapsed.
        fig_width: Width, in inches, of the figure.
        fig_height: Height, in inches, of the figure.

    Returns:
        The figure holding the decision tree plot of 'model.estimators_[0]'.
    """
    fig, ax = plt.subplots(  # pyright: ignore[reportAny]
        figsize=(fig_width, fig_height)
    )

    _ = plot_tree(
        model.estimators_[0],
        filled=True,
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=["0", "1"],
        ax=ax,  # pyright: ignore[reportAny]
    )
    ax.set_title(  # pyright: ignore[reportAny]
        f"Forest decision tree (depth limited to {max_depth})"
    )

    return fig
