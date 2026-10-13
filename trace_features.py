"""Topological and baseline features for reasoning traces."""
import numpy as np
from ripser import ripser
from scipy.stats import kurtosis, skew

GENERIC_PH_NAMES = ["h0_total", "h0_max", "h0_entropy", "h1_count", "h1_max", "h1_total", "h1_entropy"]
BETA0_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
ORIGINAL_PH_NAMES = (
    ["count_h2", "euler_char"]
    + [f"beta0_{t}" for t in BETA0_THRESHOLDS]
    + ["sh0", "sh1", "std_h0", "min_h0", "median_h0", "skew_h0", "kurt_h0", "count_h0", "maxmean_ratio_h0", "cv_h0", "iqr_h0"]
)
H0_STAT_NAMES = ["std_h0", "min_h0", "median_h0", "skew_h0", "kurt_h0", "maxmean_ratio_h0", "cv_h0", "iqr_h0"]


def cosine_distance_matrix(X):
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    D = 1.0 - Xn @ Xn.T
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0.0, 2.0)


def persistence_diagrams(D, maxdim=1):
    return ripser(np.asarray(D, dtype=np.float64), distance_matrix=True, maxdim=maxdim)["dgms"]


def _finite_lifetimes(dgm):
    dgm = np.asarray(dgm, dtype=np.float64).reshape(-1, 2)
    dgm = dgm[np.isfinite(dgm[:, 1])]
    return dgm[:, 1] - dgm[:, 0]


def persistence_entropy(lifetimes):
    lifetimes = np.asarray(lifetimes, dtype=np.float64)
    lifetimes = lifetimes[lifetimes > 0]
    if lifetimes.size == 0:
        return 0.0
    p = lifetimes / lifetimes.sum()
    return float(-(p * np.log(p)).sum())


def ph_features(D):
    """Generic PH summaries on a cosine distance matrix."""
    D = np.asarray(D, dtype=np.float64)
    if D.shape[0] < 2:
        return {name: 0.0 for name in GENERIC_PH_NAMES}
    dgms = persistence_diagrams(D, maxdim=1)
    l0 = _finite_lifetimes(dgms[0])
    l1 = _finite_lifetimes(dgms[1])
    return {
        "h0_total": float(l0.sum()),
        "h0_max": float(l0.max()) if l0.size else 0.0,
        "h0_entropy": persistence_entropy(l0),
        "h1_count": float(l1.size),
        "h1_max": float(l1.max()) if l1.size else 0.0,
        "h1_total": float(l1.sum()),
        "h1_entropy": persistence_entropy(l1),
    }


def _even_subsample(X, max_points):
    n = X.shape[0]
    if n <= max_points:
        return X
    idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
    return X[idx]


def original_ph_features(X, max_points=200):
    """Author's original feature set: ripser called on the raw point cloud (Euclidean), maxdim 2.

    Assumption: euler_char is the alternating bar count over the whole diagram, #H0 - #H1 + #H2.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        feats = {name: 0.0 for name in ORIGINAL_PH_NAMES}
        feats["count_h0"] = float(n)
        feats["euler_char"] = float(n)
        for t in BETA0_THRESHOLDS:
            feats[f"beta0_{t}"] = float(n)
        return feats
    X = _even_subsample(X, max_points)
    dgms = ripser(X, maxdim=2)["dgms"]
    h0 = np.asarray(dgms[0], dtype=np.float64).reshape(-1, 2)
    h1 = np.asarray(dgms[1], dtype=np.float64).reshape(-1, 2)
    h2 = np.asarray(dgms[2], dtype=np.float64).reshape(-1, 2)
    l0 = _finite_lifetimes(h0)
    l1 = _finite_lifetimes(h1)

    feats = {
        "count_h2": float(len(h2)),
        "euler_char": float(len(h0) - len(h1) + len(h2)),
        "count_h0": float(len(h0)),
        "sh0": persistence_entropy(l0),
        "sh1": persistence_entropy(l1),
    }
    for t in BETA0_THRESHOLDS:
        feats[f"beta0_{t}"] = float(np.sum(h0[:, 1] > t))

    if l0.size:
        mean = float(l0.mean())
        std = float(l0.std())
        q75, q25 = np.percentile(l0, [75, 25])
        feats.update({
            "std_h0": std,
            "min_h0": float(l0.min()),
            "median_h0": float(np.median(l0)),
            "skew_h0": float(skew(l0)) if l0.size >= 3 else 0.0,
            "kurt_h0": float(kurtosis(l0)) if l0.size >= 4 else 0.0,
            "maxmean_ratio_h0": float(l0.max() / mean) if mean > 0 else 0.0,
            "cv_h0": std / mean if mean > 0 else 0.0,
            "iqr_h0": float(q75 - q25),
        })
    else:
        feats.update({name: 0.0 for name in H0_STAT_NAMES})

    return {k: float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for k, v in feats.items()}


def baseline_features(X, points):
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    n_chars = float(sum(len(s) for s in points))
    if n < 2:
        return {"n_points": float(n), "n_chars": n_chars, "mean_consecutive_sim": 1.0, "mean_pairwise_dist": 0.0}
    D = cosine_distance_matrix(X)
    consecutive_sim = 1.0 - np.diag(D, k=1)
    upper = np.triu_indices(n, k=1)
    return {
        "n_points": float(n),
        "n_chars": n_chars,
        "mean_consecutive_sim": float(consecutive_sim.mean()),
        "mean_pairwise_dist": float(D[upper].mean()),
    }


PATH_PH_NAMES = [
    "path_h1_count",
    "path_h1_total",
    "path_h1_max",
    "path_h1_entropy",
    "path_h1_count_per_turn",
    "path_h1_total_per_turn",
    "path_h1_median_birth",
    "path_h1_min_birth",
]


def path_distance_matrix(X):
    """Cosine distances with consecutive turns joined at 0, so the trajectory is in the complex from the start."""
    D = cosine_distance_matrix(X)
    n = D.shape[0]
    idx = np.arange(n - 1)
    D[idx, idx + 1] = 0.0
    D[idx + 1, idx] = 0.0
    return D


def path_ph_features(X):
    """H1 of the path-augmented Rips complex: every bar is a loop closed by a return to an earlier state."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    empty = {name: 0.0 for name in PATH_PH_NAMES}
    empty["path_h1_median_birth"] = 2.0
    empty["path_h1_min_birth"] = 2.0
    if n < 4:
        return empty
    dgms = persistence_diagrams(path_distance_matrix(X), maxdim=1)
    h1 = np.asarray(dgms[1], dtype=np.float64).reshape(-1, 2)
    h1 = h1[np.isfinite(h1[:, 1])]
    lifetimes = h1[:, 1] - h1[:, 0]
    keep = lifetimes > 0
    h1 = h1[keep]
    lifetimes = lifetimes[keep]
    if lifetimes.size == 0:
        return empty
    births = h1[:, 0]
    return {
        "path_h1_count": float(lifetimes.size),
        "path_h1_total": float(lifetimes.sum()),
        "path_h1_max": float(lifetimes.max()),
        "path_h1_entropy": persistence_entropy(lifetimes),
        "path_h1_count_per_turn": float(lifetimes.size / n),
        "path_h1_total_per_turn": float(lifetimes.sum() / n),
        "path_h1_median_birth": float(np.median(births)),
        "path_h1_min_birth": float(births.min()),
    }


def recurrence_features(X, gap=3):
    """Non-topological control: share of turns that come back close to a turn at least `gap` steps earlier."""
    D = cosine_distance_matrix(X)
    n = D.shape[0]
    if n <= gap:
        return {"revisit_frac": 0.0, "revisit_frac_tight": 0.0}
    nearest_earlier = np.array([D[i, : i - gap + 1].min() for i in range(gap, n)])
    return {
        "revisit_frac": float((nearest_earlier < 0.2).mean()),
        "revisit_frac_tight": float((nearest_earlier < 0.1).mean()),
    }