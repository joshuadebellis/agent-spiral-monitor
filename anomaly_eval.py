"""Anomaly detection: score each trajectory by its distance from the shape of successful runs (per model, length-matched)."""
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from persim import PersistenceImager
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

POINT_TYPES = ["reasoning", "action", "observation"]
DIAGRAM_TYPES = ["path_h1", "rips_h0"]
VARIANTS = ["raw", "norm"]
METHODS = ["prototype", "knn"]
SUBSETS = ["all"]


def fit_imager(dgms, pixels):
    nonempty = [d for d in dgms if len(d)]
    if not nonempty:
        return None
    pts = np.vstack(nonempty).astype(np.float64)
    births = pts[:, 0]
    pers = pts[:, 1] - pts[:, 0]
    b_lo = float(births.min())
    b_span = max(float(np.quantile(births, 0.99)) - b_lo, 1e-4)
    p_span = max(float(np.quantile(pers, 0.99)), 1e-4)
    pixel = max(b_span, p_span) / pixels
    return PersistenceImager(
        birth_range=(b_lo, b_lo + pixel * pixels),
        pers_range=(0.0, pixel * pixels),
        pixel_size=pixel,
        kernel_params={"sigma": [[pixel ** 2, 0.0], [0.0, pixel ** 2]]},
    )


def to_images(dgms, imager, pixels):
    out = np.zeros((len(dgms), pixels * pixels), dtype=np.float64)
    if imager is None:
        return out
    for i, d in enumerate(dgms):
        if len(d):
            img = np.asarray(imager.transform(np.asarray(d, dtype=np.float64), skew=True), dtype=np.float64)
            flat = img.ravel()
            out[i, : flat.size] = flat[: pixels * pixels]
    return out


def normalize_rows(images):
    sums = images.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return images / sums


def anomaly_scores(images, failed, turns, groups, folds, seed, k):
    n = len(failed)
    proto = np.zeros(n)
    knn = np.zeros(n)
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in cv.split(images, failed, groups):
        tr_succ = tr[failed[tr] == 0]
        cuts = np.quantile(turns[tr_succ], [1 / 3, 2 / 3])
        band_tr = np.digitize(turns[tr_succ], cuts)
        band_te = np.digitize(turns[te], cuts)
        for b in range(3):
            idx = te[band_te == b]
            if len(idx) == 0:
                continue
            ref_idx = tr_succ[band_tr == b]
            if len(ref_idx) == 0:
                ref_idx = tr_succ
            ref = images[ref_idx]
            proto[idx] = np.linalg.norm(images[idx] - ref.mean(axis=0), axis=1)
            dists = np.sqrt(((images[idx][:, None, :] - ref[None, :, :]) ** 2).sum(axis=2))
            kk = min(k, ref.shape[0])
            knn[idx] = np.sort(dists, axis=1)[:, :kk].mean(axis=1)
    return {"prototype": proto, "knn": knn}


def subset_mask(df, subset):
    success = (df["failed"] == 0).to_numpy()
    if subset == "all":
        return np.ones(len(df), dtype=bool)
    if subset == "no_patch":
        return success | (df["no_patch"] == 1).to_numpy()
    return success | (df["wrong_patch"] == 1).to_numpy()


def combined_scores(X, y, groups, folds, seed):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]


def boot_auc_and_delta(y, score, base, combo, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs, deltas = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        aucs.append(roc_auc_score(yb, score[idx]))
        deltas.append(roc_auc_score(yb, combo[idx]) - roc_auc_score(yb, base[idx]))
    a_lo, a_hi = np.percentile(aucs, [2.5, 97.5])
    d_lo, d_hi = np.percentile(deltas, [2.5, 97.5])
    return float(a_lo), float(a_hi), float(d_lo), float(d_hi)


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config_anomaly.yaml"
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    seed = cfg["seed"]
    tag = cfg["tag"]
    data_dir = Path(cfg["data_dir"])
    out_dir = Path(cfg["results_dir"]) / f"anomaly_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pixels = int(cfg["image_pixels"])

    df_all = pd.read_parquet(data_dir / f"anomaly_{tag}.parquet")
    with open(data_dir / f"anomaly_{tag}_diagrams.pkl", "rb") as f:
        diagrams = pickle.load(f)

    records = []
    generators = [g for g, c in df_all["generator"].value_counts().items() if c >= cfg["min_generator_n"]]
    for gen in generators:
        df = df_all[df_all["generator"] == gen].reset_index(drop=True)
        failed = df["failed"].to_numpy()
        turns = df["n_turns"].to_numpy(dtype=float)
        log_turns = np.log1p(turns)
        groups = df["instance_id"].to_numpy()
        print(f"\n{gen}: {len(df)} runs, failure rate {failed.mean():.3f}, no-patch share of failures {df['no_patch'].sum() / max(failed.sum(), 1):.3f}")

        base_by_subset = {}
        for subset in SUBSETS:
            m = subset_mask(df, subset)
            base_by_subset[subset] = combined_scores(log_turns[m].reshape(-1, 1), failed[m], groups[m], cfg["cv_folds"], seed)
            records.append({
                "generator": gen, "point_type": "length", "diagram": "-", "variant": "-", "method": "turns",
                "subset": subset, "n": int(m.sum()), "auc": float(roc_auc_score(failed[m], turns[m])),
                "auc_lo": np.nan, "auc_hi": np.nan, "delta": 0.0, "delta_lo": np.nan, "delta_hi": np.nan,
            })

        combos = list(product(POINT_TYPES, DIAGRAM_TYPES, VARIANTS))
        for ptype, dtype, variant in tqdm(combos, desc=f"{gen}", unit="combo"):
            dgms = [diagrams[i][ptype][dtype] for i in df["id"]]
            imager = fit_imager(dgms, pixels)
            images = to_images(dgms, imager, pixels)
            if variant == "norm":
                images = normalize_rows(images)
            scores = anomaly_scores(images, failed, turns, groups, cfg["cv_folds"], seed, int(cfg["knn_k"]))
            for method in METHODS:
                s = scores[method]
                for subset in SUBSETS:
                    m = subset_mask(df, subset)
                    y = failed[m]
                    X = np.c_[log_turns[m], s[m]]
                    combo = combined_scores(X, y, groups[m], cfg["cv_folds"], seed)
                    base = base_by_subset[subset]
                    a_lo, a_hi, d_lo, d_hi = boot_auc_and_delta(y, s[m], base, combo, int(cfg["n_bootstrap"]), seed)
                    records.append({
                        "generator": gen, "point_type": ptype, "diagram": dtype, "variant": variant, "method": method,
                        "subset": subset, "n": int(m.sum()), "auc": float(roc_auc_score(y, s[m])),
                        "auc_lo": a_lo, "auc_hi": a_hi,
                        "delta": float(roc_auc_score(y, combo) - roc_auc_score(y, base)),
                        "delta_lo": d_lo, "delta_hi": d_hi,
                    })

    table = pd.DataFrame(records)
    table.to_csv(out_dir / "anomaly_table.csv", index=False)

    pd.set_option("display.width", 200)
    for gen in generators:
        g = table[table["generator"] == gen]
        print(f"\n=== {gen} | length baseline AUC (turns) ===")
        print(g[g["point_type"] == "length"][["subset", "n", "auc"]].to_string(index=False))
        wide = g[g["point_type"] != "length"].pivot_table(
            index=["point_type", "diagram", "variant", "method"],
            columns="subset",
            values=["auc", "delta"],
        )
        wide.columns = [f"{v}_{s}" for v, s in wide.columns]
        wide = wide[[f"{v}_{s}" for s in SUBSETS for v in ["auc", "delta"]]]
        print(f"\n=== {gen} | anomaly AUC and gain over length (delta), by subset ===")
        print(wide.round(3).sort_values("delta_no_patch", ascending=False).to_string())
        sig = g[(g["point_type"] != "length") & (g["delta_lo"] > 0)]
        print(f"\n{gen}: combinations whose gain over length has a 95% CI above zero:")
        print(sig[["point_type", "diagram", "variant", "method", "subset", "auc", "delta", "delta_lo", "delta_hi"]].round(3).to_string(index=False) if len(sig) else "  none")
    print(f"\nWrote {out_dir / 'anomaly_table.csv'}")


if __name__ == "__main__":
    main()