import numpy as np

from .. import errors

logger = errors.logger_config.LoggerSingleton().get_logger()


def build_training_table(
    weak_classifiers: list[np.ndarray],
    group_labels: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a list of per-metric similarity matrices into one feature table.

    Each 'weak_classifier' is an NxN matrix holding, for one metric and
    statistic, the similarity between every pair of audios. This function
    stacks those matrices side by side into a single table. One row per
    unordered audio pair {i, j} (i < j), one column per weak classifier, plus
    a binary label per row. That table is the direct input expected by a
    downstream classifier such as Random Forest.

    Only one row is produced per pair because the similarity matrices are
    symmetric (comparing audio i against audio j yields the same value as
    comparing j against i) so (i, j) and (j, i) always carry identical
    feature values and the same label. Emitting both as separate rows would
    not add information, it would silently duplicate observations. Also, if
    this table were ever split by row for train/test, it would risk placing
    one copy in training and its exact twin in testing. Restricting to i < j
    removes the duplication at the source. 'generate_forest' relies on this
    by calling 'build_training_table' separately on the train and test audio
    subsets, so no pair ever crosses that boundary.

    Args:
        weak_classifiers:
            List of square, equally-shaped NxN similarity matrices, one per
            metric/statistic combination.
        group_labels:
            Binary (0/1) group membership label for each of the N audios.

    Returns:
        A tuple (X, y), X has shape (n_pairs_valid, n_classifiers), y has
        shape (n_pairs_valid,). y[k] is 1 only if both audios in pair k
        belong to the labeled group. Pairs where any weak classifier holds
        NaN (a comparison failed, see task_manager._comparator) are dropped,
        so n_pairs_valid can be less than n * (n - 1) // 2.
    """
    if not weak_classifiers:
        raise ValueError("The list of weak classifiers is empty")

    first_classifier = weak_classifiers[0]
    for i, classifier in enumerate(weak_classifiers):
        if classifier.shape[0] != classifier.shape[1]:
            raise ValueError(
                f"Classifier at position {i} is not square. Its shape is {classifier.shape}."
            )
        if classifier.shape != first_classifier.shape:
            raise ValueError(
                f"Classifier at position {i} has a different shape "
                f"({classifier.shape}) than the first one, ({first_classifier.shape})"
            )

    if not all(label in {0, 1} for label in group_labels):
        raise ValueError("'group_labels' can only contain 0s and 1s")
    if len(group_labels) != first_classifier.shape[0]:
        raise ValueError(
            f"Size of 'group_labels' ({len(group_labels)}) does not match "
            f"the height of the classifier tables ({first_classifier.shape[0]})"
        )

    n = first_classifier.shape[0]
    n_pairs = n * (n - 1) // 2

    features = np.zeros((n_pairs, len(weak_classifiers)))
    labels = np.zeros(n_pairs)

    row_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for column, classifier in enumerate(weak_classifiers):
                features[row_idx, column] = classifier[i, j]
            # 1 if both audios are in the classified group, 0 otherwise.
            labels[row_idx] = group_labels[i] * group_labels[j]
            row_idx += 1

    valid = ~np.isnan(features).any(axis=1)
    if not np.all(valid):
        logger.warning(
            f"Dropped {int((~valid).sum())} pair(s) with NaN features. "
            "This is caused by a failed comparison somewhere in that pair, "
            "whether from a memory limit or any other error (see "
            "task_manager._comparator)."
        )
        features, labels = features[valid], labels[valid]

    return features, labels
