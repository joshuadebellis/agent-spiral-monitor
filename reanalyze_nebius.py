"""Re-run the nebius anomaly evaluation within a turn-count band (length-matched check)."""
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from anomaly_eval import anomaly_scores, boot_auc_and_delta, combined_scores, fit_imager, normalize_rows, to_images

POINT_TYPES = ["reasoning", "action", "observation"]
DIAGRAM_TYPES = ["path_h1", "rips_h0"]
VARIANTS = ["raw", "norm"]
METHODS = ["prototype", "knn"]
COMPARISONS = ["hit_limit", "wrong"]


def main():
    cfg_path = sys.argv[1]
    min_turns = float(sys.argv[2])
    max_turns = float(sys.argv[3])
    windows = sys.argv[4].split(",")
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    seed = cfg["seed"]
    tag = cfg["tag"]
    pixels = int(cfg["image_pixels"])
    data_dir = Path(cfg["data_dir"])
    out_dir = Path(cfg["results_dir"]) / f"anomaly_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_dir / f"anomaly_{tag}.parquet")
    with open(data_dir / f"anomaly_{tag}_diagrams.pkl", "rb") as f:
        diagrams = pickle.load(f)
    df = df[(df["n_turns"] >= min_turns) & (df["n_turns"] <= max_turns)].reset_index(drop=True)
    print(f"Band {min_turns:.0f}-{max_turns:.0f} turns, windows {windows}")
    print(df.groupby("group")["n_turns"].describe()[["count", "25%", "50%", "75%"]].to_string())

    failed = df["failed"].to_numpy()
    turns = df["n_turns"].to_numpy(dtype=float)
    log_turns = np.log1p(turns)
    groups = df["instance_id"].to_numpy()
    masks = {c: df["group"].isin(["success", c]).to_numpy() for c in COMPARISONS}

    records = []
    base = {}
    for c, m in masks.items():
        base[c] = combined_scores(log_turns[m].reshape(-1, 1), failed[m], groups[m], cfg["cv_folds"], seed)
        print(f"Length AUC within band, {c} vs success: {roc_auc_score(failed[m], turns[m]):.3f}  (n={int(m.sum())})")

    for win, ptype, dtype, variant in tqdm(list(product(windows, POINT_TYPES, DIAGRAM_TYPES, VARIANTS)), unit="combo"):
        dgms = [diagrams[i][win][ptype][dtype] for i in df["id"]]
        images = to_images(dgms, fit_imager(dgms, pixels), pixels)
        if variant == "norm":
            images = normalize_rows(images)
        scores = anomaly_scores(images, failed, turns, groups, cfg["cv_folds"], seed, int(cfg["knn_k"]))
        for method, c in product(METHODS, COMPARISONS):
            m = masks[c]
            y = failed[m]
            s = scores[method][m]
            combo = combined_scores(np.c_[log_turns[m], s], y, groups[m], cfg["cv_folds"], seed)
            a_lo, a_hi, d_lo, d_hi = boot_auc_and_delta(y, s, base[c], combo, int(cfg["n_bootstrap"]), seed)
            records.append({
                "window": win, "point_type": ptype, "diagram": dtype, "variant": variant, "method": method,
                "comparison": c, "auc": float(roc_auc_score(y, s)), "auc_lo": a_lo, "auc_hi": a_hi,
                "delta": float(roc_auc_score(y, combo) - roc_auc_score(y, base[c])), "delta_lo": d_lo, "delta_hi": d_hi,
            })

    table = pd.DataFrame(records)
    name = f"band_{int(min_turns)}_{int(max_turns)}_{'-'.join(windows)}.csv"
    table.to_csv(out_dir / name, index=False)
    pd.set_option("display.width", 250)
    print("\nTop 15 by gain over length (hit_limit vs success):")
    top = table[table["comparison"] == "hit_limit"].sort_values("delta", ascending=False).head(15)
    print(top.round(3).to_string(index=False))
    print("\nwrong vs success, top 5:")
    print(table[table["comparison"] == "wrong"].sort_values("delta", ascending=False).head(5).round(3).to_string(index=False))
    print(f"\nWrote {out_dir / name}")


if __name__ == "__main__":
    main()