"""Re-run the pre-registered tests with different CV seeds to check that the verdicts are stable."""
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from repetition_eval import ALL_LENGTHS, PREREGISTERED, run_test

SEEDS = [1, 2, 3, 4, 5]


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config_nebius_confirm.yaml"
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    tag = cfg["tag"]
    data_dir = Path(cfg["data_dir"])
    out_dir = Path(cfg["results_dir"]) / f"anomaly_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_dir / f"anomaly_{tag}.parquet")
    rep = pd.read_parquet(data_dir / f"repetition_{tag}.parquet")
    df = df.merge(rep, on="id", how="inner").reset_index(drop=True)
    with open(data_dir / f"anomaly_{tag}_diagrams.pkl", "rb") as f:
        diagrams = pickle.load(f)

    jobs = [(s, t, b) for s in SEEDS for t in PREREGISTERED for b in (t["band"], ALL_LENGTHS)]
    records = []
    for s, t, b in tqdm(jobs, desc="seeds x tests", unit="test"):
        c = dict(cfg)
        c["seed"] = s
        r = run_test(df, diagrams, t, b, c)
        r["seed"] = s
        records.append(r)

    table = pd.DataFrame(records)
    table.to_csv(out_dir / "seed_check.csv", index=False)
    summary = table.groupby(["test", "band"], sort=False).agg(
        auc_topo_mean=("auc_topo", "mean"),
        gain_rep_mean=("gain_vs_len_rep", "mean"),
        gain_rep_min=("gain_vs_len_rep", "min"),
        gain_rep_lo_min=("gain_vs_len_rep_lo", "min"),
        passes=("verdict", lambda v: f"{int((v == 'PASS').sum())}/{len(v)}"),
    ).reset_index()
    pd.set_option("display.width", 250)
    print("\nStability across 5 CV seeds (gain = adding topology to length + repetition):")
    print(summary.round(3).to_string(index=False))
    print(f"\nWrote {out_dir / 'seed_check.csv'}")


if __name__ == "__main__":
    main()