"""Re-stream the exact rows used by build_nebius_anomaly.py and add plain repetition features (no topology) per window."""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset
from tqdm import tqdm

from build_nebius_anomaly import POINT_TYPES, WINDOWS, traj_hash, turns_from_trajectory, window_indices

WS = re.compile(r"\s+")
WORD = re.compile(r"\w+")
REP_FEATS = ["uniq_frac", "top_frac", "period2_frac", "consec_jaccard"]


def norm_text(s):
    return WS.sub(" ", s or "").strip().lower()


def jaccard(a, b):
    sa, sb = set(WORD.findall(a)), set(WORD.findall(b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def repetition_features(texts):
    t = [norm_text(s) for s in texts]
    n = len(t)
    counts = Counter(t)
    period2 = [t[i] == t[i - 2] for i in range(2, n)]
    consec = [jaccard(t[i - 1], t[i]) for i in range(1, n)]
    return {
        "uniq_frac": len(counts) / n,
        "top_frac": max(counts.values()) / n,
        "period2_frac": float(np.mean(period2)) if period2 else 0.0,
        "consec_jaccard": float(np.mean(consec)) if consec else 0.0,
    }


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config_nebius.yaml"
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    seed = cfg["seed"]
    tag = cfg["tag"]
    w = int(cfg["window"])
    data_dir = Path(cfg["data_dir"])

    df = pd.read_parquet(data_dir / f"anomaly_{tag}.parquet")
    wanted = {
        int(row_id.rsplit("#", 1)[1]): (iid, int(n))
        for row_id, iid, n in zip(df["id"], df["instance_id"], df["n_turns"])
    }
    last_pos = max(wanted)

    ds = load_dataset(cfg["dataset"], split=cfg["split"], streaming=True)
    if cfg.get("stream_skip"):
        ds = ds.skip(int(cfg["stream_skip"]))
    if cfg.get("shuffle_buffer"):
        ds = ds.shuffle(seed=seed, buffer_size=int(cfg["shuffle_buffer"]))

    rows = []
    hashes = []
    mismatched = 0
    scanned = 0
    with tqdm(total=len(wanted), desc=f"re-streaming {tag}", unit="traj") as bar:
        for ex in ds:
            scanned += 1
            if scanned > last_pos:
                break
            if scanned not in wanted:
                continue
            iid, n_expected = wanted[scanned]
            turns = turns_from_trajectory(ex["trajectory"], cfg)
            bar.update(1)
            if ex["instance_id"] != iid or len(turns) != n_expected:
                mismatched += 1
                continue
            idx = window_indices(len(turns), w)
            rec = {"id": f"{iid}#{scanned}"}
            for win in WINDOWS:
                for ptype in POINT_TYPES:
                    feats = repetition_features([turns[i][ptype] for i in idx[win]])
                    for name, value in feats.items():
                        rec[f"{win}_{ptype}_{name}"] = value
            rows.append(rec)
            hashes.append(traj_hash(ex["trajectory"]))

    matched = len(rows)
    print(f"\nMatched {matched} of {len(wanted)} rows, mismatched {mismatched}.")
    if matched < 0.9 * len(wanted):
        raise SystemExit("Fewer than 90% of rows matched: the stream order is not reproducible. Stop and tell Claude.")

    rep = pd.DataFrame(rows)
    rep.to_parquet(data_dir / f"repetition_{tag}.parquet", index=False)
    (data_dir / f"seen_{tag}.txt").write_text("\n".join(hashes), encoding="utf-8")

    merged = df.merge(rep, on="id", how="inner")
    cols = [f"last_action_{f}" for f in REP_FEATS]
    print("\nLast-window action repetition, median by group:")
    print(merged.groupby("group")[cols].median().round(3).to_string())
    print(f"\nWrote {data_dir / f'repetition_{tag}.parquet'} and {data_dir / f'seen_{tag}.txt'}")


if __name__ == "__main__":
    main()