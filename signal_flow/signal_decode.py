"""
Frame-id decoding for Allen movie embeddings (same logic as Flow_Hackathon/notebooks/get_signal_dataset.ipynb).

Use modality=\"ca\" for calcium embeddings (no temporal pooling) or \"neuropixel\" (factor-4 pooling).
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def allen_frame_id_decode(
    train_fs,
    train_labels,
    test_fs,
    test_labels,
    modality: str = "neuropixel",
    decoder: str = "knn",
):
    if modality == "neuropixel":
        factor = 4
    elif modality == "ca":
        factor = 1
    else:
        raise ValueError(modality)

    time_window = 1

    def feature_for_one_frame(feature):
        arr = np.asarray(feature)
        return arr.reshape(-1, factor, arr.shape[-1]).mean(axis=1)

    train_fs = feature_for_one_frame(train_fs)
    test_fs = feature_for_one_frame(test_fs)

    if decoder != "knn":
        raise ValueError("This helper only implements decoder='knn'.")

    params = np.power(np.linspace(1, 10, 5, dtype=int), 2)
    errs = []

    for n in params:
        train_decoder = KNeighborsClassifier(n_neighbors=n, metric="cosine")
        train_valid_idx = int(len(train_fs) / 9 * 8)
        train_decoder.fit(train_fs[:train_valid_idx], train_labels[:train_valid_idx])
        pred = train_decoder.predict(train_fs[train_valid_idx:])
        err = train_labels[train_valid_idx:] - pred
        errs.append(abs(err).sum())

    test_decoder = KNeighborsClassifier(
        n_neighbors=params[np.argmin(errs)], metric="cosine"
    )
    test_decoder.fit(train_fs, train_labels)
    pred = test_decoder.predict(test_fs)
    frame_errors = pred - test_labels

    def _quantize_acc(frame_diff, time_window=1):
        true = (abs(frame_diff) < (time_window * 30)).sum()
        return true / len(frame_diff) * 100

    quantized_acc = _quantize_acc(frame_errors, time_window)
    return pred, frame_errors, quantized_acc
