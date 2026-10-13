"""Pre-registered tests: does the topology anomaly score add beyond length AND plain repetition features?"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from add_repetition import REP_FEATS
from anomaly_eval import anomaly_scores, boot_auc_and_delta, combined_scores, fit_imager, normalize_rows, to_images
from build_nebius_anomaly import POINT_TYPES

PREREGISTERED = [
    {"name": "P1_late_collapse", "window": "last", "point_type": "action", "diagram": "rips_h0",
     "variant": "norm", "method": "knn", "comparison": "hit_limit", "band": (31, 69)},
    {"name": "P2_midrun_reasoning_loops", "window": "even", "point_type": "reasoning", "diagram": "path_h1",
     "variant": "norm", "method": "knn", "comparison": "hit_limit", "band": (31, 69)},
    {"name": "P3_early_warning", "window": "first", "point_type": "action", "diagram": "path_h1",
     "variant": "norm", "method": "knn", "comparison": "hit_limit", "band": (30, 100000)},
    {"name": "P4_wrong_answers", "window": "last", "point_type": "reasoning", "diagram": "rips_h0",
     "variant": "raw", "method": "prototype", "comparison": "wrong", "band": (31, 69)},
]
ALL_LENGTHS = (15, 100000)


def run_test(df_all, diagrams, test, band, cfg):
    seed = cfg["seed"]
    folds = cfg["cv_folds"]
    pixels = int(cfg["image_pixels"])
    n_boot = int(cfg["n_bootstrap"])
    lo, hi = band
    df = df_all[(df_all["n_turns"] >= lo) & (df_all["n_turns"] <= hi)].reset_index(drop=True)

    failed = df["failed"].to_numpy()
    turns = df["n_turns"].to_numpy(dtype=float)
    groups = df["instance_id"].to_numpy()

    dgms = [diagrams[i][test["window"]][test["point_type"]][test["diagram"]] for i in df["id"]]
    images = to_images(dgms, fit_imager(dgms, pixels), pixels)
    if test["variant"] == "norm":
        images = normalize_rows(images)
    scores = anomaly_scores(images, failed, turns, groups, folds, seed, int(cfg["knn_k"]))

    m = df["group"].isin(["success", test["comparison"]]).to_numpy()
    y = failed[m]
    g = groups[m]
    s = scores[test["method"]][m]
    L = np.log1p(turns[m]).reshape(-1, 1)
    rep_cols = [f"{test['window']}_{p}_{f}" for p in POINT_TYPES for f in REP_FEATS]
    R = df.loc[m, rep_cols].to_numpy(dtype=float)

    base_len = combined_scores(L, y, g, folds, seed)
    full_len = combined_scores(np.c_[L, s], y, g, folds, seed)
    base_rep = combined_scores(np.c_[L, R], y, g, folds, seed)
    full_rep = combined_scores(np.c_[L, R, s], y, g, folds, seed)

    a_lo, a_hi, dl_lo, dl_hi = boot_auc_and_delta(y, s, base_len, full_len, n_boot, seed)
    _, _, dr_lo, dr_hi = boot_auc_and_delta(y, s, base_rep, full_rep, n_boot, seed)
    delta_rep = float(roc_auc_score(y, full_rep) - roc_auc_score(y, base_rep))
    return {
        "test": test["name"],
        "band": f"{lo}-{hi}" if hi < 100000 else f"{lo}+",
        "n": int(m.sum()),
        "n_success": int((y == 0).sum()),
        "auc_topo": float(roc_auc_score(y, s)),
        "auc_topo_lo": a_lo,
        "auc_topo_hi": a_hi,
        "auc_len": float(roc_auc_score(y, base_len)),
        "auc_len_rep": float(roc_auc_score(y, base_rep)),
        "auc_len_rep_topo": float(roc_auc_score(y, full_rep)),
        "gain_vs_len": float(roc_auc_score(y, full_len) - roc_auc_score(y, base_len)),
        "gain_vs_len_lo": dl_lo,
        "gain_vs_len_hi": dl_hi,
        "gain_vs_len_rep": delta_rep,
        "gain_vs_len_rep_lo": dr_lo,
        "gain_vs_len_rep_hi": dr_hi,
        "verdict": "PASS" if dr_lo > 0 else "fail",
    }


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config_nebius.yaml"
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    tag = cfg["tag"]
    data_dir = Path(cfg["data_dir"])
    out_dir = Path(cfg["results_dir"]) / f"anomaly_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_dir / f"anomaly_{tag}.parquet")
    rep = pd.read_parquet(data_dir / f"repetition_{tag}.parquet")
    with open(data_dir / f"anomaly_{tag}_diagrams.pkl", "rb") as f:
        diagrams = pickle.load(f)
    n_before = len(df)
    df = df.merge(rep, on="id", how="inner").reset_index(drop=True)
    print(f"{tag}: {len(df)} of {n_before} rows have repetition features.")

    jobs = [(t, t["band"]) for t in PREREGISTERED] + [(t, ALL_LENGTHS) for t in PREREGISTERED]
    records = [run_test(df, diagrams, t, b, cfg) for t, b in tqdm(jobs, desc="pre-registered tests", unit="test")]
    table = pd.DataFrame(records)
    table.to_csv(out_dir / "preregistered.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n=== AUCs (topology score alone; length; length + repetition; all three) ===")
    print(table[["test", "band", "n", "n_success", "auc_topo", "auc_topo_lo", "auc_topo_hi",
                 "auc_len", "auc_len_rep", "auc_len_rep_topo"]].round(3).to_string(index=False))
    print("\n=== Gain from adding the topology score (95% bootstrap CI) ===")
    print(table[["test", "band", "gain_vs_len", "gain_vs_len_lo", "gain_vs_len_hi",
                 "gain_vs_len_rep", "gain_vs_len_rep_lo", "gain_vs_len_rep_hi", "verdict"]].round(3).to_string(index=False))
    print("\nPASS = gain over length + repetition has a 95% CI above zero.")
    if tag == "nebius_main":
        print("NOTE: these four tests were chosen on nebius_main, so results here are exploratory. Only the confirm run counts.")
    print(f"\nWrote {out_dir / 'preregistered.csv'}")


if __name__ == "__main__":
    main()