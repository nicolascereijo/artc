import itertools

import numpy as np
import numpy.typing as npt

from .. import errors

logger = errors.logger_config.LoggerSingleton().get_logger()


def build_training_table(
    weak_classifiers: list[npt.NDArray[np.float64]],
    group_labels: list[int],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Flattens per-metric similarity matrices into one feature table.

    Each 'weak_classifier' is an NxN matrix holding, for one metric and
    statistic, the similarity between every pair of audios. This function
    stacks those matrices side by side into a single table, with one row per
    unordered audio pair '(i, j)' where 'i < j', one column per weak classifier
    and a binary label per row.

    Only one row is produced per pair because the similarity matrices are
    symmetric, so '(i, j)' and '(j, i)' always carry the same feature values
    and label. Restricting to 'i < j' keeps that duplication out of the table
    from the start, which matters because 'generate_forest' builds the train
    and test tables separately, on the train and test audio subsets, so a
    duplicated pair can never end up split across both.

    Args:
        weak_classifiers: List of square 'NxN' similarity matrices of equal
            shape, one per metric and statistic combination.
        group_labels: Binary (0/1) group membership label for each of the 'N'
            audios.

    Returns:
        A tuple '(features, labels)'. 'features' has shape '(n_pairs_valid,
        n_classifiers)' and 'labels' has shape '(n_pairs_valid,)'. A label is
        '1' only when both audios in that pair belong to the labeled group.
        Pairs where any weak classifier holds 'NaN' (a comparison failed, see
        'task_manager._comparator') are dropped, so 'n_pairs_valid' can be less
        than 'n * (n - 1) // 2'.

    Raises:
        ValueError: If 'weak_classifiers' is empty, if any classifier matrix is
            not square or its shape does not match the others, if
            'group_labels' holds a value other than '0' or '1', or if its
            length does not match the classifier matrices' size.
    """
    if not weak_classifiers:
        raise ValueError("The list of weak classifiers is empty")

    first_classifier = weak_classifiers[0]
    for i, classifier in enumerate(weak_classifiers):
        if classifier.shape[0] != classifier.shape[1]:
            raise ValueError(
                f"Classifier at position {i} is not square, its shape is " +
                f"{classifier.shape}"
            )
        if classifier.shape != first_classifier.shape:
            raise ValueError(
                f"Classifier at position {i} has a different shape " +
                f"({classifier.shape}) than the first one " +
                f"({first_classifier.shape})"
            )

    if not all(label in {0, 1} for label in group_labels):
        raise ValueError("'group_labels' can only contain 0s and 1s")
    if len(group_labels) != first_classifier.shape[0]:
        raise ValueError(
            f"Size of 'group_labels' ({len(group_labels)}) does not match " +
            "the height of the classifier tables " +
            f"({first_classifier.shape[0]})"
        )

    n = first_classifier.shape[0]
    # Total unordered pairs among 'n' audios, 'n' choose 2.
    n_pairs = n * (n - 1) // 2

    features = np.zeros((n_pairs, len(weak_classifiers)))
    labels = np.zeros(n_pairs)

    pairs = itertools.combinations(range(n), 2)
    for row_idx, (i, j) in enumerate(pairs):
        for column, classifier in enumerate(weak_classifiers):
            features[row_idx, column] = classifier[i, j]
        # The label is '1' only when both audios belong to the labeled group,
        # '0' otherwise.
        labels[row_idx] = group_labels[i] * group_labels[j]

    # A pair is invalid when any weak classifier feature is 'NaN'.
    invalid: npt.NDArray[np.bool_] = (  # pyright: ignore[reportAny]
        np.isnan(features).any(axis=1)
    )
    if invalid.any():
        dropped = int(invalid.sum())  # pyright: ignore[reportAny]
        logger.warning(
            f"Dropped {dropped} pair(s) with 'NaN' features, caused by a " +
            "failed comparison somewhere in that pair, whether from a " +
            "memory limit or any other error (see " +
            "'task_manager._comparator')"
        )
        features, labels = features[~invalid], labels[~invalid]

    return features, labels
