from typing import Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.extmath import weighted_mode

from .math import sigmoid


def k_neighbors_classify(X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_test: np.ndarray,
                         n_neighbors: int = 20,
                         similarity=cosine_similarity) -> Tuple[np.ndarray, np.ndarray]:
    preds = np.zeros(len(X_test), dtype=int)
    scores = np.zeros(len(X_test), dtype=float)
    sim = sigmoid(similarity(X_test, X_train))

    for i in range(len(X_test)):
        candidates = np.argsort(sim[i])[-n_neighbors:]
        labels = y_train[candidates]

        weights_ = sim[i][candidates]
        mode, score = weighted_mode(labels, weights_)

        preds[i] = int(mode[0])
        scores[i] = score[0]

    return preds, scores


def k_neighbors_classify_scores(X_train: np.ndarray,
                                y_train: np.ndarray,
                                X_test: np.ndarray,
                                n_neighbors: int = 20,
                                similarity=cosine_similarity) -> Tuple[np.ndarray, np.ndarray]:
    real_labels, y_train = np.unique(y_train, return_inverse=True)
    scores = np.zeros((len(X_test), len(real_labels)), dtype=float)
    sim = sigmoid(similarity(X_test, X_train))

    for i in range(len(X_test)):
        candidates = np.argsort(sim[i])[-n_neighbors:]

        for it in candidates:
            scores[i][y_train[it]] += sim[i][it]

    return scores, real_labels
