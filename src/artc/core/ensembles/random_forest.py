from collections.abc import Mapping, Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

from .. import charts
from .dataset import build_training_table

ClassWeight = (
    Literal["balanced", "balanced_subsample"]
    | Mapping[int, float]
    | Sequence[Mapping[int, float]]
    | None
)


def generate_forest(
    weak_classifiers: list[npt.NDArray[np.float64]],
    classifier_names: list[str],
    group_labels: list[int],
    n_splits: int = 5,
    *,
    class_weight: ClassWeight = "balanced",
    min_samples_leaf: int = 3,
    max_depth: int | None = None,
):
    """Cross validates a Random Forest over the audio pair similarity table
    and reports the results.

    A row level train test split leaks audio identity, since every audio
    appears in 'n_audios' minus one pair rows. A random split of rows almost
    always puts some of an audio's rows in training and the rest in test, so
    the model can learn that audio's traits from training and recognize it in a
    nominally unseen test row.

    To avoid that, this function splits by audio identity into 'n_splits'
    folds, stratified on 'group_labels' so every fold keeps a similar class
    ratio and none ends up without positive audios. Since each audio is its own
    group, this guarantees no audio is ever split across train and test. Each
    fold then calls 'build_training_table' separately on its train and test
    audio submatrices, so no pair can mix a train audio with a test audio.

    One 'RandomForestClassifier' is trained per fold, and its classification
    report is printed. Positive pairs are scarce, since forming one needs two
    positive audios in the same fold, so a single fold's test side often holds
    only a few, or none. Averaging F1 per fold would be dominated by that near
    binary noise, so accuracy and class 1 F1 are instead computed once, pooling
    every fold's test predictions together, alongside the raw TP/FP/FN counts.

    'min_samples_leaf' and 'class_weight' default to mild regularization and
    rebalancing. Without them, the forest has enough capacity, given how few
    positive pairs there are, to fit the training positives perfectly while
    still assigning a low probability to the positive pair held out for
    testing, memorizing instead of generalizing.

    Diagnostic plots (confusion matrix, ROC curve, feature importance, decision
    tree) are generated only for the last fold's model, using
    'classifier_names' to label each feature column. They are written as
    'confusion_matrix.png', 'roc_curve.png', 'feature_importance.png' and
    'decision_tree.png' in the current working directory, overwriting any file
    already there with that name.

    Args:
        weak_classifiers: List of square 'NxN' similarity matrices of equal
            shape, one per metric and statistic combination.
        classifier_names: Readable name for each weak classifier, in the same
            order as 'weak_classifiers'. Used to label the feature importance
            plot.
        group_labels: Binary (0/1) group membership label for each of the 'N'
            audios.
        n_splits: Number of stratified folds over audio identities.
        class_weight: Forwarded to 'RandomForestClassifier'. Reweights the
            scarce positive pair class during training.
        min_samples_leaf: Forwarded to 'RandomForestClassifier'. Keeps a single
            leaf from being carved out for one training pair.
        max_depth: Forwarded to 'RandomForestClassifier'. 'None' keeps
            sklearn's default of unrestricted depth.

    Raises:
        ValueError: If 'n_splits' is too high for the number of audios in
            'group_labels', leaving the smallest fold with fewer than 2 audios.
    """
    n_audios = len(group_labels)
    if n_audios // n_splits < 2:
        raise ValueError(
            f"n_splits={n_splits} is too high for {n_audios} audios, the " +
            "smallest fold would end up with fewer than 2 audios, leaving " +
            f"no pairs to test on, use n_splits <= {n_audios // 2}"
        )

    audio_ids = np.arange(n_audios)
    folds = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=42,
    ).split(audio_ids, group_labels)

    def table_for(
        ids: npt.NDArray[np.int64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        # Slices each NxN matrix down to one side's audios before pairing
        # them up, so a pair can never mix a train audio with a test audio.
        classifiers = [c[np.ix_(ids, ids)] for c in weak_classifiers]
        labels = [
            group_labels[i] for i in ids.tolist()  # pyright: ignore[reportAny]
        ]
        features, y = build_training_table(classifiers, labels)
        return features, y.astype(np.int64)

    # Pooled across every fold's test predictions, not averaged per fold, see
    # the docstring for why the F1 per fold is too noisy at this sample size.
    y_test_folds: list[npt.NDArray[np.int64]] = []
    y_pred_folds: list[npt.NDArray[np.int64]] = []
    model = X_test = y_test = None

    for fold_idx, (train_ids, test_ids) in enumerate(folds, start=1):
        X_train, y_train = table_for(train_ids)
        X_test, y_test = table_for(test_ids)

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight=class_weight,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
        )
        model.fit(X_train, y_train)
        y_pred: npt.NDArray[np.int64] = model.predict(X_test).astype(np.int64)

        print(f"--- Fold {fold_idx}/{n_splits} ---")
        print(classification_report(
            y_test,
            y_pred,
            zero_division=0,  # pyright: ignore[reportArgumentType]
        ))
        has_positive: bool = (y_test == 1).any()  # pyright: ignore[reportAny]
        if not has_positive:
            print(f"Fold {fold_idx}: no positive pairs in test side")
        y_test_folds.append(y_test)
        y_pred_folds.append(y_pred)

    y_test_all: npt.NDArray[np.int64] = np.concatenate(y_test_folds)
    y_pred_all: npt.NDArray[np.int64] = np.concatenate(y_pred_folds)

    is_tp: npt.NDArray[np.bool_] = (  # pyright: ignore[reportAny]
        (y_pred_all == 1) & (y_test_all == 1)
    )
    is_fp: npt.NDArray[np.bool_] = (  # pyright: ignore[reportAny]
        (y_pred_all == 1) & (y_test_all == 0)
    )
    is_fn: npt.NDArray[np.bool_] = (  # pyright: ignore[reportAny]
        (y_pred_all == 0) & (y_test_all == 1)
    )
    tp = int(is_tp.sum())  # pyright: ignore[reportAny]
    fp = int(is_fp.sum())  # pyright: ignore[reportAny]
    fn = int(is_fn.sum())  # pyright: ignore[reportAny]
    n_positive = tp + fn

    print(
        f"Pooled over {n_splits} folds ({n_positive} positive pairs total): " +
        f"TP={tp} FP={fp} FN={fn}"
    )
    print(f"Pooled accuracy: {accuracy_score(y_test_all, y_pred_all):.3f}")
    f1 = f1_score(
        y_test_all,
        y_pred_all,
        pos_label=1,
        zero_division=0,  # pyright: ignore[reportArgumentType]
    )
    print(f"Pooled F1 (class 1): {f1:.3f}")

    # Diagnostic plots use the last fold's model and test set. 'model',
    # 'X_test' and 'y_test' still hold them since the loop above never rescopes
    # them. The assert below is for the type checker, a 'StratifiedKFold' with
    # 'n_splits' of 2 or more always yields at least one fold, so these are
    # never actually 'None'.
    assert model is not None and X_test is not None and y_test is not None
    figs = charts.tree_plots(model, classifier_names, X_test, y_test)
    names = (
        "confusion_matrix", "roc_curve", "feature_importance", "decision_tree",
    )
    for fig, name in zip(figs, names):
        fig.savefig(f"{name}.png")
        # Releases each figure once saved, instead of leaving it in
        # matplotlib's global registry for the rest of the process's lifetime.
        plt.close(fig)
