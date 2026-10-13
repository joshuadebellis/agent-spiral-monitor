"""Anomaly scores vs successful runs, per fixed window: hit_limit vs success and wrong vs success."""
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

WINDOWS = ["first", "last", "even"]
POINT_TYPES = ["reasoning", "action", "observation"]
DIAGRAM_TYPES = ["path_h1", "rips_h0"]
VARIANTS = ["raw", "norm"]
METHODS = ["prototype", "knn"]
COMPARISONS = ["hit_limit", "wrong"]


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config_nebius.yaml"
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

    failed = df["failed"].to_numpy()
    turns = df["n_turns"].to_numpy(dtype=float)
    log_turns = np.log1p(turns)
    groups = df["instance_id"].to_numpy()
    masks = {c: df["group"].isin(["success", c]).to_numpy() for c in COMPARISONS}

    records = []
    base = {}
    for c, m in masks.items():
        base[c] = combined_scores(log_turns[m].reshape(-1, 1), failed[m], groups[m], cfg["cv_folds"], seed)
        records.append({
            "window": "-", "point_type": "length", "diagram": "-", "variant": "-", "method": "turns",
            "comparison": c, "n": int(m.sum()), "auc": float(roc_auc_score(failed[m], turns[m])),
            "auc_lo": np.nan, "auc_hi": np.nan, "delta": 0.0, "delta_lo": np.nan, "delta_hi": np.nan,
        })

    combos = list(product(WINDOWS, POINT_TYPES, DIAGRAM_TYPES, VARIANTS))
    for win, ptype, dtype, variant in tqdm(combos, desc="window x point x diagram x variant", unit="combo"):
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
                "comparison": c, "n": int(m.sum()), "auc": float(roc_auc_score(y, s)),
                "auc_lo": a_lo, "auc_hi": a_hi,
                "delta": float(roc_auc_score(y, combo) - roc_auc_score(y, base[c])),
                "delta_lo": d_lo, "delta_hi": d_hi,
            })

    table = pd.DataFrame(records)
    table.to_csv(out_dir / "anomaly_table.csv", index=False)

    pd.set_option("display.width", 250)
    print("\n=== Length baseline (AUC of total turns) ===")
    print(table[table["point_type"] == "length"][["comparison", "n", "auc"]].to_string(index=False))
    wide = table[table["point_type"] != "length"].pivot_table(
        index=["window", "point_type", "diagram", "variant", "method"],
        columns="comparison",
        values=["auc", "delta"],
    )
    wide.columns = [f"{v}_{c}" for v, c in wide.columns]
    wide = wide[[f"{v}_{c}" for c in COMPARISONS for v in ["auc", "delta"]]]
    print("\n=== Anomaly AUC and gain over length (AUC < 0.5 means failures look MORE typical than successes) ===")
    print(wide.round(3).sort_values("delta_hit_limit", ascending=False).to_string())
    sig = table[(table["point_type"] != "length") & (table["delta_lo"] > 0)]
    print("\nCombinations whose gain over length has a 95% CI above zero:")
    print(sig[["window", "point_type", "diagram", "variant", "method", "comparison", "auc", "delta", "delta_lo", "delta_hi"]].round(3).to_string(index=False) if len(sig) else "  none")
    print(f"\nWrote {out_dir / 'anomaly_table.csv'}")


if __name__ == "__main__":
    main()