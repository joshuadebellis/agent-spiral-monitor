"""Embed each reasoning step, sentence or word and compute baseline, generic PH and original PH features."""
import pickle
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from trace_features import baseline_features, cosine_distance_matrix, original_ph_features, ph_features

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\\(\[$])|\n+")
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")


def split_sentences(text):
    parts = [p.strip() for p in SENTENCE_SPLIT.split(text)]
    return [p for p in parts if len(p) >= 3]


def to_points(steps, granularity):
    if granularity == "step":
        return steps
    if granularity == "sentence":
        return [sentence for step in steps for sentence in split_sentences(step)]
    if granularity == "word":
        return [word for step in steps for word in WORD_PATTERN.findall(step)]
    raise ValueError(f"unknown granularity {granularity}")


def even_subsample(X, max_points):
    n = X.shape[0]
    if n <= max_points:
        return X
    idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
    return X[idx]


def main():
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    granularity = cfg["granularity"]
    tag = cfg["tag"]
    limit = cfg.get("limit_per_split")
    max_points = cfg["max_points_ph"]
    data_dir = Path(cfg["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(cfg["embedding_model"], device=cfg["device"])

    rows = []
    traces = {}
    skipped = 0
    for split in cfg["splits"]:
        ds = load_dataset(cfg["dataset"], split=split)
        if limit:
            ds = ds.select(range(min(int(limit), len(ds))))
        for ex in tqdm(ds, desc=f"{split}: embed + PH ({tag})", unit="trace"):
            steps = [s for s in ex["steps"] if s.strip()]
            points = to_points(steps, granularity)
            if not points:
                skipped += 1
                continue
            X = model.encode(
                points,
                batch_size=256,
                normalize_embeddings=cfg["normalize_embeddings"],
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            X_ph = even_subsample(X, max_points)
            rows.append({
                "id": ex["id"],
                "split": split,
                "generator": ex["generator"],
                "final_answer_correct": bool(ex["final_answer_correct"]),
                "label": int(ex["label"]),
                "n_steps": float(len(steps)),
                **baseline_features(X, points),
                **ph_features(cosine_distance_matrix(X_ph)),
                **original_ph_features(X_ph, max_points),
            })
            traces[ex["id"]] = {"steps": steps, "X": X_ph.astype(np.float32)}

    df = pd.DataFrame(rows)
    df.to_parquet(data_dir / f"features_{tag}.parquet", index=False)
    with open(data_dir / f"traces_{tag}.pkl", "wb") as f:
        pickle.dump(traces, f)

    norms = np.linalg.norm(X, axis=1)
    print(f"\nSaved {len(df)} traces ({skipped} skipped as empty), tag '{tag}', to {data_dir}")
    print(f"Embedding norm check on last trace: min {norms.min():.3f}  max {norms.max():.3f}  (all 1.000 means the model normalizes)")
    summary = df.groupby("split").agg(
        n=("id", "size"),
        frac_final_wrong=("final_answer_correct", lambda s: float((~s).mean())),
        median_steps=("n_steps", "median"),
        median_points=("n_points", "median"),
        frac_with_h1_loop=("h1_count", lambda s: float((s > 0).mean())),
        frac_with_h2_void=("count_h2", lambda s: float((s > 0).mean())),
        median_euler=("euler_char", "median"),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()