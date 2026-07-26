import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from .. import charts
from .dataset import build_training_table


def preprocessing(
    weak_classifiers: list[np.ndarray], group_labels: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the training table and split it into train/test sets

    Delegates to `build_training_table` for the feature/label table (one row
    per unique audio pair) and then applies a standard 80/20 train/test
    split. The split uses a fixed random seed so that repeated runs over the
    same input data produce the same partition, matching the fixed seed
    already used for `RandomForestClassifier` in `generate_forest`.

    Args:
        weak_classifiers:
            List of square, equally-shaped NxN similarity matrices, one per
            metric/statistic combination
        group_labels:
            Binary (0/1) group membership label for each of the N audios

    Returns:
        (X_train, X_test, y_train, y_test), the feature/label arrays for
        each split
    """
    features, labels = build_training_table(weak_classifiers, group_labels)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)
    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train),
        np.asarray(y_test),
    )


def generate_forest(
    weak_classifiers: list[np.ndarray], classifier_names: list[str], group_labels: list[int]
):
    """Train a Random Forest on the audio-pair similarity table and report results.

    Builds the train/test split via `preprocessing`, fits a
    `RandomForestClassifier` on the training set, and prints a classification
    report (precision/recall/F1) evaluated on the held-out test set. It also
    saves four diagnostic plots (confusion matrix, ROC curve, feature
    importance and a sample decision tree) to the current working directory,
    using `classifier_names` to label which metric/statistic each feature
    column corresponds to.

    Args:
        weak_classifiers:
            List of square, equally-shaped NxN similarity matrices, one per
            metric/statistic combination.
        classifier_names:
            Human-readable name for each weak classifier, in the same order
            as `weak_classifiers`. Used to label feature importance plots.
        group_labels:
            Binary (0/1) group membership label for each of the N audios.
    """
    X_train, X_test, y_train, y_test = preprocessing(weak_classifiers, group_labels)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

    fig1, fig2, fig3, fig4 = charts.tree_plots(model, classifier_names, X_test, y_test)
    fig1.savefig("confusion_matrix.png")
    fig2.savefig("roc_curve.png")
    fig3.savefig("feature_importance.png")
    fig4.savefig("decision_tree.png")
