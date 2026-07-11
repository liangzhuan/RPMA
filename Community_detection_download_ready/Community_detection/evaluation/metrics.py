import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
)


def clustering_accuracy(y_true, y_pred):
    """Clustering accuracy after best label permutation via Hungarian matching."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    true_classes = np.unique(y_true)
    pred_classes = np.unique(y_pred)
    true_map = {c: i for i, c in enumerate(true_classes)}
    pred_map = {c: i for i, c in enumerate(pred_classes)}

    cm = np.zeros((len(true_classes), len(pred_classes)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[true_map[t], pred_map[p]] += 1

    # Maximize matched count = minimize negative count.
    row_ind, col_ind = linear_sum_assignment(-cm)
    return cm[row_ind, col_ind].sum() / y_true.size


def evaluate(y_true, y_pred):
    """Return common clustering metrics."""
    return {
        "ACC": clustering_accuracy(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "ARI": adjusted_rand_score(y_true, y_pred),
    }
