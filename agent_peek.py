"""Peek at Open-SWE-Traces without downloading the full dataset."""
import json
from collections import Counter

from datasets import load_dataset
from tqdm import tqdm

ds = load_dataset("nvidia/Open-SWE-Traces", "v1.0", split="openhands", streaming=True)

ex = next(iter(ds))
print("fields:", list(ex.keys()))
print("resolved:", ex["resolved"], "| language:", ex["language"], "| n_messages:", len(ex["messages"]))
print("metadata keys:", list(ex["metadata"].keys()))
print("teacher_model:", json.dumps(ex["metadata"].get("teacher_model")))
print("\nFirst 6 messages (role, keys, first 400 chars):")
for i, m in enumerate(ex["messages"][:6]):
    print(f"[{i}] role={m.get('role')} keys={list(m.keys())}")
    print("    ", json.dumps(m)[:400])

resolved_counts = Counter()
role_counts = Counter()
teacher_counts = Counter()
for row in tqdm(ds.take(2000), total=2000, desc="tally first 2000", unit="traj"):
    resolved_counts[row["resolved"]] += 1
    teacher = row["metadata"].get("teacher_model") or {}
    teacher_counts[teacher.get("name") if isinstance(teacher, dict) else str(teacher)] += 1
    for m in row["messages"]:
        role_counts[m.get("role")] += 1
print("\nresolved counts (first 2000):", dict(resolved_counts))
print("teacher models (first 2000):", dict(teacher_counts))
print("message roles (first 2000):", dict(role_counts))