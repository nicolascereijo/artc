import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold

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

    A row-level train/test split (e.g. plain 'train_test_split' on
    build_training_table's output) leaks audio identity. Every audio
    participates in n_audios-1 pair-rows, so a random split of rows almost
    always leaves some of an audio's rows in training and some in test,
    letting the model learn that audio's idiosyncratic features and
    "recognize" it in a nominally unseen row. To avoid that, audio
    identities (not rows) are split into 'n_splits' GroupKFold folds first.
    Each fold then builds its train/test tables by calling
    'build_training_table' separately on the train-audios submatrix and the
    test-audios submatrix, so no pair can ever mix a train audio with a test
    audio.

    Trains one 'RandomForestClassifier' per fold and prints its
    classification report, then prints the mean +/- std of accuracy and
    class-1 F1 across folds, since any single fold's test set is small and
    noisy on its own. Diagnostic plots (confusion matrix, ROC curve, feature
    importance, sample decision tree) are saved only for the last fold's
    model, using `classifier_names` to label which metric/statistic each
    feature column corresponds to.

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
            Number of GroupKFold folds over audio identities.
    """
    n_audios = len(group_labels)
    if n_audios // n_splits < 2:
        raise ValueError(
            f"n_splits={n_splits} is too high for {n_audios} audios: the "
            f"smallest fold would end up with fewer than 2 audios, leaving "
            f"no pairs to test on. Use n_splits <= {n_audios // 2}."
        )

    audio_ids = np.arange(n_audios)
    folds = GroupKFold(n_splits=n_splits, shuffle=True, random_state=42).split(
        audio_ids, groups=audio_ids
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
        f1s_class1.append(f1_score(y_test, y_pred, pos_label=1, zero_division=0))  # pyright: ignore[reportArgumentType]

    print(f"Mean accuracy across {n_splits} folds: {np.mean(accuracies):.3f} "
          f"(+/- {np.std(accuracies):.3f})")
    print(f"Mean F1 (class 1) across {n_splits} folds: {np.mean(f1s_class1):.3f} "
          f"(+/- {np.std(f1s_class1):.3f})")

    # Diagnostic plots use the last fold's model/test set - `model`/`X_test`/
    # `y_test` still hold it since the loop above doesn't rescope them. The
    # assert is for the type checker: GroupKFold(n_splits>=2) always yields
    # at least one fold, so these are never actually None.
    assert model is not None and X_test is not None and y_test is not None
    fig1, fig2, fig3, fig4 = charts.tree_plots(model, classifier_names, X_test, y_test)
    fig1.savefig("confusion_matrix.png")
    fig2.savefig("roc_curve.png")
    fig3.savefig("feature_importance.png")
    fig4.savefig("decision_tree.png")
