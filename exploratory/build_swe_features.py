"""Build PH + baseline features for Open-SWE-Traces agent trajectories (one point per assistant turn)."""
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
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from trace_features import baseline_features, cosine_distance_matrix, original_ph_features, ph_features


def text_of(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def assistant_turn_text(message, max_reasoning_chars, max_action_chars):
    thought = (text_of(message.get("reasoning_content")) + " " + text_of(message.get("content"))).strip()
    actions = []
    for call in message.get("tool_calls") or []:
        fn = (call or {}).get("function") or {}
        args = fn.get("arguments") or ""
        if not isinstance(args, str):
            args = json.dumps(args)
        actions.append(f"[{fn.get('name', '')}] {args}")
    action = " ".join(actions)
    return f"{thought[:max_reasoning_chars]}\n{action[:max_action_chars]}".strip()


def even_subsample(X, max_points):
    n = X.shape[0]
    if n <= max_points:
        return X
    idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
    return X[idx]


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config_swe.yaml"
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tag = cfg["tag"]
    data_dir = Path(cfg["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    max_points = int(cfg["max_points_ph"])
    target_n = int(cfg["max_trajectories"])

    ds = load_dataset(cfg["dataset"], cfg["dataset_config"], split=cfg["dataset_split"], streaming=True)
    if cfg.get("shuffle_buffer"):
        ds = ds.shuffle(seed=seed, buffer_size=int(cfg["shuffle_buffer"]))
    model = SentenceTransformer(cfg["embedding_model"], device=cfg["device"])

    rows = []
    traces = {}
    seen = 0
    skipped_unlabeled = 0
    skipped_short = 0
    with tqdm(total=target_n, desc=f"trajectories ({tag})", unit="traj") as bar:
        for ex in ds:
            seen += 1
            if ex["resolved"] not in (0, 1):
                skipped_unlabeled += 1
                continue
            points = [
                assistant_turn_text(m, cfg["max_reasoning_chars"], cfg["max_action_chars"])
                for m in ex["messages"]
                if m.get("role") == "assistant"
            ]
            points = [p for p in points if p]
            if len(points) < 2:
                skipped_short += 1
                continue
            X = model.encode(
                points,
                batch_size=128,
                normalize_embeddings=cfg["normalize_embeddings"],
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            X_ph = even_subsample(X, max_points)
            metadata = ex.get("metadata") or {}
            teacher = metadata.get("teacher_model") or {}
            rows.append({
                "id": ex["trajectory_id"],
                "instance_id": ex["instance_id"],
                "repo": ex["repo"],
                "split": ex["language"],
                "category": metadata.get("category", ""),
                "generator": teacher.get("name", "unknown") if isinstance(teacher, dict) else str(teacher),
                "final_answer_correct": bool(ex["resolved"] == 1),
                "label": -1 if ex["resolved"] == 1 else 0,
                "n_steps": float(len(points)),
                "n_messages": float(len(ex["messages"])),
                **baseline_features(X, points),
                **ph_features(cosine_distance_matrix(X_ph)),
                **original_ph_features(X_ph, max_points),
            })
            traces[ex["trajectory_id"]] = {"steps": points, "X": X_ph.astype(np.float32)}
            bar.update(1)
            if len(rows) >= target_n:
                break

    df = pd.DataFrame(rows)
    df.to_parquet(data_dir / f"features_{tag}.parquet", index=False)
    with open(data_dir / f"traces_{tag}.pkl", "wb") as f:
        pickle.dump(traces, f)

    print(f"\nStreamed {seen} rows: kept {len(df)}, skipped {skipped_unlabeled} unlabeled and {skipped_short} with <2 assistant turns")
    print(f"Unique tasks (instance_id): {df['instance_id'].nunique()}   resolved rate: {df['final_answer_correct'].mean():.3f}")
    summary = df.groupby("split").agg(
        n=("id", "size"),
        tasks=("instance_id", "nunique"),
        resolved_rate=("final_answer_correct", "mean"),
        median_turns=("n_steps", "median"),
        frac_with_h1_loop=("h1_count", lambda s: float((s > 0).mean())),
        frac_with_h2_void=("count_h2", lambda s: float((s > 0).mean())),
    ).sort_values("n", ascending=False)
    print(summary.to_string())
    print("\nMedian assistant turns by outcome:")
    print(df.groupby("final_answer_correct")["n_steps"].median().to_string())


if __name__ == "__main__":
    main()