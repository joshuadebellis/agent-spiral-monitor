"""Peek at nebius/SWE-agent-trajectories without downloading everything."""
import json
from collections import Counter

from datasets import load_dataset
from tqdm import tqdm

ds = load_dataset("nebius/SWE-agent-trajectories", split="train", streaming=True)

ex = next(iter(ds))
print("fields:", list(ex.keys()))
print("model:", ex["model_name"], "| target:", ex["target"], "| exit_status:", ex["exit_status"], "| n_steps:", len(ex["trajectory"]))
print("\nFirst 5 trajectory entries (keys and first 300 chars):")
for i, step in enumerate(ex["trajectory"][:5]):
    keys = list(step.keys()) if isinstance(step, dict) else type(step).__name__
    print(f"[{i}] keys={keys}")
    print("    ", json.dumps(step)[:300])

counts = Counter()
for row in tqdm(ds.take(3000), total=3000, desc="tally", unit="traj"):
    counts[(row["model_name"], row["target"], row["exit_status"])] += 1
print("\n(model, resolved, exit_status) -> count, first 3000 rows:")
for key, c in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {key}: {c}")