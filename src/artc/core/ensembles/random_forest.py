from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from .. import charts

def preprocessing(weak_classifiers: list[np.ndarray], group_labels: list[int]) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first_classifier = weak_classifiers[0]
    binary_labels = []

    if not weak_classifiers:
        raise ValueError("The list of weak classifiers is empty")
    for i, classifier in enumerate(weak_classifiers):
        if classifier.shape[0] != classifier.shape[1]:
            raise ValueError(
                f"Classifier at position {i} is not square. Its shape is {classifier.shape}.")
        if classifier.shape != first_classifier.shape:
            raise ValueError(
                f"Classifier at position {i} has a different shape ({
                    classifier.shape}) than the first one, ({first_classifier.shape})")

    if not all(label in {0, 1} for label in group_labels):
        raise ValueError("'group_labels' can only contain 0s and 1s")
    if len(group_labels) != first_classifier.shape[0]:
        raise ValueError(
            f"Size of 'binary_labels' ({
                len(group_labels)}) does not match the height of the classifier tables ({
                    first_classifier.shape[0]})")

    # Rows equal to the number of audio pair combinations minus the number of equal pairs
    dataset_rows = sum(classifier.shape[0] * (classifier.shape[1] - 1) for classifier in weak_classifiers)
    dataset = np.zeros((dataset_rows, len(weak_classifiers)))

    row_idx = 0
    for column, classifier in enumerate(weak_classifiers):
        for i in range(classifier.shape[0]): # Classifier rows
            for j in range(classifier.shape[1]): # Classifier columns
                if i != j: # Discard values from the diagonal
                    dataset[row_idx, column] = classifier[i, j]
                    # 1 if both audios are in the classified group, 0 otherwise
                    binary_labels.append(group_labels[i] * group_labels[j])
                    row_idx += 1

    # Merge the data with labels, then shuffle and split into two sets
    dataset = np.hstack((dataset, np.array(binary_labels).reshape(-1, 1)))
    np.random.shuffle(dataset)

    train_size = int(0.8 * len(dataset))
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]

    # Separate the labels from the data
    train_labels = train_data[:, -1]
    test_labels = test_data[:, -1]
    train_data = train_data[:, :-1]
    test_data = test_data[:, :-1]

    return train_data, test_data, train_labels, test_labels


def generate_forest(weak_classifiers: list[np.ndarray], classifier_names: list[str],
    group_labels: list[int]):
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
