import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

from .. import charts
from .dataset import build_training_table


def generate_forest(
    weak_classifiers: list[np.ndarray],
    classifier_names: list[str],
    group_labels: list[int],
    n_splits: int = 5,
):
    """Cross-validate a Random Forest on the audio-pair similarity table and
    report results.

    A row level train test split (e.g. plain 'train_test_split' on
    build_training_table's output) leaks audio identity. Every audio
    participates in n_audios minus one pair rows, so a random split of rows
    almost always leaves some of an audio's rows in training and some in
    test, letting the model learn that audio's idiosyncratic features and
    recognize it in a nominally unseen row. To avoid that, audio identities,
    not rows, are split into 'n_splits' folds first. Each audio is its own
    group of size one at this granularity, so a plain per audio split,
    stratified on 'group_labels' to keep the class ratio in every fold and
    avoid folds whose test side has no positive labeled audios at all,
    already guarantees no audio is ever split across train and test. Each
    fold then builds its train and test tables by calling
    'build_training_table' separately on the train audios submatrix and the
    test audios submatrix, so no pair can ever mix a train audio with a
    test audio.

    Trains one 'RandomForestClassifier' per fold and prints its
    classification report. A fold whose test side ends up with zero
    positive pairs (both audios sharing the labeled group) can't score a
    meaningful F1 for the positive class, since it would report 0.0
    regardless of how good the model is, given there is nothing of that
    class to recognize. Such folds are excluded from the F1 average, though
    still included in the accuracy average and still fully printed, and
    are noted explicitly. Diagnostic plots (confusion matrix, ROC curve,
    feature importance, sample decision tree) are saved only for the last
    fold's model, using `classifier_names` to label which metric and
    statistic each feature column corresponds to.

    Args:
        weak_classifiers:
            List of square, equally-shaped NxN similarity matrices, one per
            metric/statistic combination.
        classifier_names:
            Human-readable name for each weak classifier, in the same order
            as 'weak_classifiers'. Used to label feature importance plots.
        group_labels:
            Binary (0/1) group membership label for each of the N audios.
        n_splits:
            Number of stratified folds over audio identities.
    """
    n_audios = len(group_labels)
    if n_audios // n_splits < 2:
        raise ValueError(
            f"n_splits={n_splits} is too high for {n_audios} audios: the "
            f"smallest fold would end up with fewer than 2 audios, leaving "
            f"no pairs to test on. Use n_splits <= {n_audios // 2}."
        )

    audio_ids = np.arange(n_audios)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(
        audio_ids, group_labels
    )

    def table_for(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Slice each NxN matrix down to one side's audios before pairing them
        # up, so a pair can never mix a train audio with a test audio.
        classifiers = [c[np.ix_(ids, ids)] for c in weak_classifiers]
        labels = [group_labels[i] for i in ids]
        features, y = build_training_table(classifiers, labels)
        return features, y.astype(int)

    # A single fold's score is noisy given how few audios end up in its test
    # side. Averaging across folds is what actually reflects generalization.
    accuracies, f1s_class1 = [], []
    model = X_test = y_test = None

    for fold_idx, (train_ids, test_ids) in enumerate(folds, start=1):
        X_train, y_train = table_for(train_ids)
        X_test, y_test = table_for(test_ids)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"--- Fold {fold_idx}/{n_splits} ---")
        print(classification_report(y_test, y_pred, zero_division=0))  # pyright: ignore[reportArgumentType]
        accuracies.append(accuracy_score(y_test, y_pred))
        if (y_test == 1).any():
            f1s_class1.append(f1_score(y_test, y_pred, pos_label=1, zero_division=0))  # pyright: ignore[reportArgumentType]
        else:
            print(f"Fold {fold_idx}: no positive pairs in test side, excluded from the F1 average")

    print(f"Mean accuracy across {n_splits} folds: {np.mean(accuracies):.3f} "
          f"(+/- {np.std(accuracies):.3f})")
    print(f"Mean F1 (class 1) across {len(f1s_class1)}/{n_splits} folds: {np.mean(f1s_class1):.3f} "
          f"(+/- {np.std(f1s_class1):.3f})")

    # Diagnostic plots use the last fold's model/test set - `model`/`X_test`/
    # `y_test` still hold it since the loop above doesn't rescope them. The
    # assert is for the type checker: StratifiedKFold(n_splits>=2) always
    # yields at least one fold, so these are never actually None.
    assert model is not None and X_test is not None and y_test is not None
    figs = charts.tree_plots(model, classifier_names, X_test, y_test)
    names = ("confusion_matrix", "roc_curve", "feature_importance", "decision_tree")
    for fig, name in zip(figs, names):
        fig.savefig(f"{name}.png")
        # Release each figure once saved, instead of leaving it in
        # matplotlib's global registry for the rest of the process's lifetime.
        plt.close(fig)
