"""Exploratory (decided after seeing the Rips results): path-augmented complex + recurrence control."""
import pickle
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from trace_features import path_ph_features, recurrence_features


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config_swe_path.yaml"
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    data_dir = Path(cfg["data_dir"])
    source_tag = cfg["source_tag"]
    tag = cfg["tag"]

    df = pd.read_parquet(data_dir / f"features_{source_tag}.parquet")
    with open(data_dir / f"traces_{source_tag}.pkl", "rb") as f:
        traces = pickle.load(f)

    rows = []
    for trajectory_id in tqdm(df["id"], desc="path complex + recurrence", unit="traj"):
        X = traces[trajectory_id]["X"]
        rows.append({"id": trajectory_id, **path_ph_features(X), **recurrence_features(X)})

    out = df.merge(pd.DataFrame(rows), on="id", how="left")
    out.to_parquet(data_dir / f"features_{tag}.parquet", index=False)
    shutil.copyfile(data_dir / f"traces_{source_tag}.pkl", data_dir / f"traces_{tag}.pkl")

    print(f"\nWrote {data_dir / f'features_{tag}.parquet'} ({len(out)} trajectories)")
    cols = ["path_h1_count", "path_h1_count_per_turn", "path_h1_total_per_turn", "path_h1_median_birth", "revisit_frac", "revisit_frac_tight"]
    print("\nMedians by outcome (False = failed):")
    print(out.groupby("final_answer_correct")[cols].median().T.to_string())


if __name__ == "__main__":
    main()