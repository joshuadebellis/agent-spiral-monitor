"""Stream nebius/SWE-agent-trajectories (one model); save fixed-window diagrams per point type and outcome group."""
import hashlib
import pickle
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from ripser import ripser
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from trace_features import cosine_distance_matrix, path_distance_matrix

POINT_TYPES = ["reasoning", "action", "observation"]
WINDOWS = ["first", "last", "even"]
CODE_BLOCK = re.compile(r"```[a-zA-Z]*\n?(.*?)```", re.DOTALL)
KEEP_GROUPS = ["success", "hit_limit", "wrong"]


def outcome_group(target, exit_status):
    if target:
        return "success" if exit_status == "submitted" else "success_other"
    if "exit_context" in exit_status:
        return "hit_limit"
    if exit_status == "submitted":
        return "wrong"
    return "other_fail"


def traj_hash(trajectory):
    h = hashlib.md5()
    for step in trajectory:
        h.update((step.get("role") or "").encode("utf-8"))
        h.update(b"\x1f")
        h.update((step.get("text") or "").encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def split_ai_text(text):
    blocks = CODE_BLOCK.findall(text or "")
    action = blocks[-1].strip() if blocks else ""
    reasoning = CODE_BLOCK.sub(" ", text or "").strip()
    return reasoning, action


def turns_from_trajectory(trajectory, cfg):
    turns = []
    for i, step in enumerate(trajectory):
        if step.get("role") != "ai":
            continue
        reasoning, action = split_ai_text(step.get("text"))
        observation = ""
        if i + 1 < len(trajectory) and trajectory[i + 1].get("role") == "user":
            observation = (trajectory[i + 1].get("text") or "").strip()
        turns.append({
            "reasoning": reasoning[: cfg["max_reasoning_chars"]] or "(no reasoning)",
            "action": action[: cfg["max_action_chars"]] or "(no action)",
            "observation": observation[: cfg["max_observation_chars"]] or "(no output)",
        })
    return turns


def window_indices(n, w):
    return {
        "first": list(range(w)),
        "last": list(range(n - w, n)),
        "even": [int(i) for i in np.linspace(0, n - 1, w).round().astype(int)],
    }


def diagrams_for(X):
    path_h1 = ripser(path_distance_matrix(X), distance_matrix=True, maxdim=1)["dgms"][1]
    rips_h0 = ripser(cosine_distance_matrix(X), distance_matrix=True, maxdim=0)["dgms"][0]
    path_h1 = np.asarray(path_h1, dtype=np.float32).reshape(-1, 2)
    rips_h0 = np.asarray(rips_h0, dtype=np.float32).reshape(-1, 2)
    rips_h0 = rips_h0[np.isfinite(rips_h0[:, 1])]
    return {"path_h1": path_h1, "rips_h0": rips_h0}


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config_nebius.yaml"
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tag = cfg["tag"]
    w = int(cfg["window"])
    cap = int(cfg["max_per_group"])
    data_dir = Path(cfg["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(cfg["dataset"], split=cfg["split"], streaming=True)
    if cfg.get("stream_skip"):
        ds = ds.skip(int(cfg["stream_skip"]))
    if cfg.get("shuffle_buffer"):
        ds = ds.shuffle(seed=seed, buffer_size=int(cfg["shuffle_buffer"]))
    model = SentenceTransformer(cfg["embedding_model"], device=cfg["device"])

    exclude = set()
    for p in cfg.get("exclude_hashes") or []:
        exclude.update(Path(p).read_text(encoding="utf-8").split())
    if exclude:
        print(f"Excluding {len(exclude)} trajectories already used in earlier runs.")

    rows = []
    diagrams = {}
    kept = {g: 0 for g in KEEP_GROUPS}
    skipped = {"other_model": 0, "other_group": 0, "group_full": 0, "too_short": 0, "seen_before": 0}
    scanned = 0
    with tqdm(total=cap * len(KEEP_GROUPS), desc=f"trajectories ({tag})", unit="traj") as bar:
        for ex in ds:
            scanned += 1
            if scanned > int(cfg["max_rows_scan"]):
                break
            if ex["model_name"] != cfg["model_name"]:
                skipped["other_model"] += 1
                continue
            group = outcome_group(bool(ex["target"]), ex["exit_status"])
            if group not in KEEP_GROUPS:
                skipped["other_group"] += 1
                continue
            if exclude and traj_hash(ex["trajectory"]) in exclude:
                skipped["seen_before"] += 1
                continue
            if kept[group] >= cap:
                skipped["group_full"] += 1
                if all(kept[g] >= cap for g in KEEP_GROUPS):
                    break
                continue
            turns = turns_from_trajectory(ex["trajectory"], cfg)
            n = len(turns)
            if n < w:
                skipped["too_short"] += 1
                continue

            idx = window_indices(n, w)
            texts = []
            slices = {}
            for win in WINDOWS:
                for t in POINT_TYPES:
                    start = len(texts)
                    texts.extend(turns[i][t] for i in idx[win])
                    slices[(win, t)] = (start, len(texts))
            X_all = model.encode(
                texts,
                batch_size=256,
                normalize_embeddings=cfg["normalize_embeddings"],
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            row_id = f"{ex['instance_id']}#{scanned}"
            diagrams[row_id] = {
                win: {t: diagrams_for(X_all[slices[(win, t)][0]: slices[(win, t)][1]]) for t in POINT_TYPES}
                for win in WINDOWS
            }
            rows.append({
                "id": row_id,
                "instance_id": ex["instance_id"],
                "group": group,
                "exit_status": ex["exit_status"],
                "failed": int(group != "success"),
                "n_turns": float(n),
                "n_entries": float(len(ex["trajectory"])),
            })
            kept[group] += 1
            bar.update(1)

    df = pd.DataFrame(rows)
    df.to_parquet(data_dir / f"anomaly_{tag}.parquet", index=False)
    with open(data_dir / f"anomaly_{tag}_diagrams.pkl", "wb") as f:
        pickle.dump(diagrams, f)

    print(f"\nScanned {scanned} rows. Kept: {kept}. Skipped: {skipped}")
    print(df.groupby("group")["n_turns"].describe()[["count", "min", "25%", "50%", "75%", "max"]].to_string())
    print(f"\nUnique tasks: {df['instance_id'].nunique()}")


if __name__ == "__main__":
    main()