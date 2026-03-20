"""
FDA paper-style evaluation (Wang et al., ICML 2025; official code: github.com/wangpuli/FDA).

The spike experiments report **R² (%)** = 100 × sklearn's `r2_score` on **continuous behavior**
(2D hand velocity in the paper). On Allen movie embeddings we use a **2D phase coding** of
frame index (sin/cos) as the regression target — same dimensionality and linear decode spirit
as `experiment/ft_func.py` (linear readout after flow / features).

Also supports **few-trial calibration**: subsample the target-area training bank by
`calibration_ratio` (default 0.02 matches `tgt_train_ratio` in `test_FDA_on_spikes_ft.py`)
before fitting decoders, optionally averaged over `calibration_seeds`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from signal_decode import allen_frame_id_decode


def frames_to_phase_behavior(frames: np.ndarray, n_stimuli: int = 900) -> np.ndarray:
    """
    Map discrete frame indices to 2D continuous targets (analogous to 2D velocity).
    theta = 2π * frame / n_stimuli; target = [sin(theta), cos(theta)].
    """
    f = np.asarray(frames, dtype=np.float64)
    theta = 2.0 * np.pi * f / float(n_stimuli)
    return np.column_stack([np.sin(theta), np.cos(theta)])


def r2_behavior_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² (%) as in the FDA README table (100 × uniform-average R² across output dims).
    Matches `r2_score(...); print("%.2f" % (r2*100))` style reporting.
    """
    return float(
        r2_score(y_true, y_pred, multioutput="uniform_average") * 100.0
    )


def linear_behavior_r2_percent(
    train_x: np.ndarray,
    train_frames: np.ndarray,
    test_x: np.ndarray,
    test_frames: np.ndarray,
    *,
    n_stimuli: int = 900,
    ridge_alpha: float = 1.0,
) -> float:
    """Ridge regression from neural features to 2D phase behavior; return R² (%)."""
    y_tr = frames_to_phase_behavior(train_frames, n_stimuli=n_stimuli)
    y_te = frames_to_phase_behavior(test_frames, n_stimuli=n_stimuli)
    reg = Ridge(alpha=ridge_alpha).fit(train_x, y_tr)
    pred = reg.predict(test_x)
    return r2_behavior_percent(y_te, pred)


def _calibration_indices(
    n_train: int, calibration_ratio: float, seed: int
) -> np.ndarray:
    """Subsample indices for few-trial calibration (at least 1 trial)."""
    rng = np.random.default_rng(seed)
    k = max(1, int(round(float(calibration_ratio) * n_train)))
    k = min(k, n_train)
    return rng.choice(n_train, size=k, replace=False)


def knn_quantized_accuracy_percent(
    train_fs: np.ndarray,
    train_labels: np.ndarray,
    test_fs: np.ndarray,
    test_labels: np.ndarray,
    *,
    modality: str = "ca",
) -> float:
    """Wrapper: existing Allen kNN frame-ID accuracy (quantized, % correct)."""
    _, _, acc = allen_frame_id_decode(
        train_fs,
        train_labels,
        test_fs,
        test_labels,
        modality=modality,
        decoder="knn",
    )
    return float(acc)


def evaluate_with_calibration(
    full_train_x: np.ndarray,
    full_train_labels: np.ndarray,
    test_x: np.ndarray,
    test_labels: np.ndarray,
    *,
    calibration_ratio: float = 1.0,
    calibration_seed: int = 0,
    n_stimuli: int = 900,
    ridge_alpha: float = 1.0,
    modality: str = "ca",
) -> Dict[str, float]:
    """
    Subsample training bank, then compute:
      - `r2_pct`: linear Ridge → phase behavior, R² (%)
      - `knn_acc_pct`: existing cosine kNN quantized frame accuracy
    """
    n = full_train_x.shape[0]
    idx = _calibration_indices(n, calibration_ratio, calibration_seed)
    tr_x = full_train_x[idx]
    tr_y = full_train_labels[idx]

    r2 = linear_behavior_r2_percent(
        tr_x, tr_y, test_x, test_labels, n_stimuli=n_stimuli, ridge_alpha=ridge_alpha
    )
    knn_acc = knn_quantized_accuracy_percent(
        tr_x, tr_y, test_x, test_labels, modality=modality
    )
    return {
        "n_calib": float(len(idx)),
        "r2_pct": r2,
        "knn_acc_pct": knn_acc,
    }


def fda_metrics_sweep(
    full_train_x: np.ndarray,
    full_train_labels: np.ndarray,
    test_cases: Sequence[Tuple[str, np.ndarray]],
    test_labels: np.ndarray,
    *,
    calibration_ratios: Sequence[float] = (0.02, 0.05, 0.1, 1.0),
    calibration_seeds: Sequence[int] = (0, 1, 2),
    n_stimuli: int = 900,
    ridge_alpha: float = 1.0,
    modality: str = "ca",
) -> List[Dict[str, object]]:
    """
    Few-trial sweep matching FDA `tgt_train_ratio` spirit: for each calibration ratio,
    average metrics over `calibration_seeds`.

    For each seed we draw **one** subsample of the training bank and evaluate **all**
    test feature sets on the same calibration set (fair comparison).

    `test_cases`: list of (name, test_features), e.g. ``("oracle", target_te_np)``,
    ``("flow", z_flow_test)``.
    """
    rows: List[Dict[str, object]] = []
    names = [n for n, _ in test_cases]

    for ratio in calibration_ratios:
        agg: Dict[str, Dict[str, List[float]]] = {
            name: {"r2": [], "knn": []} for name in names
        }
        n_calib_last = 0

        for seed in calibration_seeds:
            n = full_train_x.shape[0]
            idx = _calibration_indices(n, ratio, seed)
            n_calib_last = len(idx)
            tr_x = full_train_x[idx]
            tr_y = full_train_labels[idx]

            for name, test_x in test_cases:
                r2 = linear_behavior_r2_percent(
                    tr_x,
                    tr_y,
                    test_x,
                    test_labels,
                    n_stimuli=n_stimuli,
                    ridge_alpha=ridge_alpha,
                )
                knn = knn_quantized_accuracy_percent(
                    tr_x, tr_y, test_x, test_labels, modality=modality
                )
                agg[name]["r2"].append(r2)
                agg[name]["knn"].append(knn)

        row: Dict[str, object] = {
            "calibration_ratio": ratio,
            "n_calib_trials": n_calib_last,
        }
        for name in names:
            r2s = agg[name]["r2"]
            knns = agg[name]["knn"]
            row[f"{name}_r2_pct_mean"] = float(np.mean(r2s))
            row[f"{name}_r2_pct_std"] = float(np.std(r2s)) if len(r2s) > 1 else 0.0
            row[f"{name}_knn_acc_mean"] = float(np.mean(knns))
        rows.append(row)
    return rows


def print_fda_sweep_table(
    rows: Sequence[Dict[str, object]],
    method_names: Sequence[str],
) -> None:
    """Pretty-print sweep results."""
    print("\n=== FDA-style evaluation (R² % + kNN acc %, few-trial calibration) ===")
    print(
        "R²: Ridge linear readout → sin/cos(frame) (2D behavior, cf. 2D velocity in the paper)."
    )
    print(
        "Few-trial: subsample target-area train bank (cf. tgt_train_ratio in FDA ft script).\n"
    )
    for row in rows:
        r = row["calibration_ratio"]
        n = row.get("n_calib_trials", "?")
        print(f"\n-- calibration_ratio={r}  (~{n} train trials) --")
        for m in method_names:
            r2m = row.get(f"{m}_r2_pct_mean", float("nan"))
            r2s = row.get(f"{m}_r2_pct_std", 0.0)
            knn = row.get(f"{m}_knn_acc_mean", float("nan"))
            if np.isnan(r2m):
                continue
            if r2s > 0:
                print(
                    f"  {m:16s}  R² % = {r2m:6.2f} ± {r2s:4.2f}   |   kNN acc % = {knn:6.2f}"
                )
            else:
                print(f"  {m:16s}  R² % = {r2m:6.2f}          |   kNN acc % = {knn:6.2f}")
