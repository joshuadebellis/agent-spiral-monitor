"""Stream Open-SWE-Traces; save path-complex H1 and Rips H0 diagrams per point type, plus failure-type metadata."""
import json
import pickle
import random
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


def text_of(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def reasoning_text(message, max_chars):
    return (text_of(message.get("reasoning_content")) + " " + text_of(message.get("content"))).strip()[:max_chars]


def action_text(message, max_chars):
    parts = []
    for call in message.get("tool_calls") or []:
        fn = (call or {}).get("function") or {}
        args = fn.get("arguments") or ""
        if not isinstance(args, str):
            args = json.dumps(args)
        parts.append(f"[{fn.get('name', '')}] {args}")
    return " ".join(parts).strip()[:max_chars]


def turn_points(messages, cfg):
    points = {t: [] for t in POINT_TYPES}
    n_turns = 0
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        n_turns += 1
        r = reasoning_text(m, cfg["max_reasoning_chars"])
        a = action_text(m, cfg["max_action_chars"])
        observed = []
        j = i + 1
        while j < len(messages) and messages[j].get("role") == "tool":
            observed.append(text_of(messages[j].get("content")))
            j += 1
        o = " ".join(observed).strip()[: cfg["max_observation_chars"]]
        if r:
            points["reasoning"].append(r)
        if a:
            points["action"].append(a)
        if o:
            points["observation"].append(o)
    return points, n_turns


def cap_points(X, max_points):
    n = X.shape[0]
    if n <= max_points:
        return X
    idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
    return X[idx]


def diagrams_for(X, max_points):
    empty = np.zeros((0, 2), dtype=np.float32)
    if X.shape[0] < 4:
        return {"path_h1": empty, "rips_h0": empty}
    Xc = cap_points(X, max_points)
    path_h1 = ripser(path_distance_matrix(Xc), distance_matrix=True, maxdim=1)["dgms"][1]
    rips_h0 = ripser(cosine_distance_matrix(Xc), distance_matrix=True, maxdim=0)["dgms"][0]
    path_h1 = np.asarray(path_h1, dtype=np.float32).reshape(-1, 2)
    rips_h0 = np.asarray(rips_h0, dtype=np.float32).reshape(-1, 2)
    rips_h0 = rips_h0[np.isfinite(rips_h0[:, 1])]
    return {"path_h1": path_h1, "rips_h0": rips_h0}


def patch_info(metadata):
    model_patch = metadata.get("model_patch")
    lines = None
    if isinstance(model_patch, dict):
        patch = model_patch.get("patch") or ""
        lines = model_patch.get("num_modified_lines")
    elif isinstance(model_patch, str):
        patch = model_patch
    else:
        patch = ""
    has_patch = bool(patch and patch.strip())
    return has_patch, (float(lines) if lines is not None else float("nan"))


def save(rows, diagrams, data_dir, tag):
    pd.DataFrame(rows).to_parquet(data_dir / f"anomaly_{tag}.parquet", index=False)
    with open(data_dir / f"anomaly_{tag}_diagrams.pkl", "wb") as f:
        pickle.dump(diagrams, f)


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config_anomaly.yaml"
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tag = cfg["tag"]
    data_dir = Path(cfg["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    allowed = set(cfg["generators"])
    target_n = int(cfg["max_trajectories"])

    ds = load_dataset(cfg["dataset"], cfg["dataset_config"], split=cfg["dataset_split"], streaming=True)
    if cfg.get("stream_skip"):
        ds = ds.skip(int(cfg["stream_skip"]))
    if cfg.get("shuffle_buffer"):
        ds = ds.shuffle(seed=seed, buffer_size=int(cfg["shuffle_buffer"]))
    model = SentenceTransformer(cfg["embedding_model"], device=cfg["device"])

    rows = []
    diagrams = {}
    seen = 0
    skipped = {"unlabeled": 0, "other_generator": 0, "too_short": 0}
    with tqdm(total=target_n, desc=f"trajectories ({tag})", unit="traj") as bar:
        for ex in ds:
            seen += 1
            if ex["resolved"] not in (0, 1):
                skipped["unlabeled"] += 1
                continue
            metadata = ex.get("metadata") or {}
            teacher = metadata.get("teacher_model") or {}
            generator = teacher.get("name", "unknown") if isinstance(teacher, dict) else str(teacher)
            if generator not in allowed:
                skipped["other_generator"] += 1
                continue
            points, n_turns = turn_points(ex["messages"], cfg)
            if len(points["reasoning"]) < 4 and len(points["action"]) < 4:
                skipped["too_short"] += 1
                continue

            all_texts = points["reasoning"] + points["action"] + points["observation"]
            X_all = model.encode(
                all_texts,
                batch_size=128,
                normalize_embeddings=cfg["normalize_embeddings"],
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            per_type = {}
            start = 0
            for t in POINT_TYPES:
                end = start + len(points[t])
                per_type[t] = diagrams_for(X_all[start:end], int(cfg["max_points_path"]))
                start = end

            has_patch, patch_lines = patch_info(metadata)
            failed = ex["resolved"] == 0
            rows.append({
                "id": ex["trajectory_id"],
                "instance_id": ex["instance_id"],
                "repo": ex["repo"],
                "split": ex["language"],
                "category": metadata.get("category", ""),
                "generator": generator,
                "failed": int(failed),
                "no_patch": int(failed and not has_patch),
                "wrong_patch": int(failed and has_patch),
                "patch_lines": patch_lines,
                "n_turns": float(n_turns),
                "n_messages": float(len(ex["messages"])),
                "n_chars": float(sum(len(s) for s in points["reasoning"] + points["action"])),
                **{f"n_points_{t}": float(len(points[t])) for t in POINT_TYPES},
            })
            diagrams[ex["trajectory_id"]] = per_type
            bar.update(1)
            if len(rows) % int(cfg["checkpoint_every"]) == 0:
                save(rows, diagrams, data_dir, tag)
            if len(rows) >= target_n:
                break

    save(rows, diagrams, data_dir, tag)
    df = pd.DataFrame(rows)
    print(f"\nStreamed {seen} rows, kept {len(df)}. Skipped: {skipped}")
    summary = df.groupby("generator").agg(
        n=("id", "size"),
        tasks=("instance_id", "nunique"),
        failure_rate=("failed", "mean"),
        no_patch_share=("no_patch", "mean"),
        wrong_patch_share=("wrong_patch", "mean"),
        median_turns=("n_turns", "median"),
        median_action_points=("n_points_action", "median"),
        median_observation_points=("n_points_observation", "median"),
    )
    print(summary.to_string())
    print("\nMedian turns by outcome type:")
    outcome = np.where(df["failed"] == 0, "success", np.where(df["no_patch"] == 1, "fail_no_patch", "fail_wrong_patch"))
    print(df.assign(outcome=outcome).groupby(["generator", "outcome"])["n_turns"].median().to_string())


if __name__ == "__main__":
    main()