"""Check patch fields and how trajectories end, to find a real 'never converged' failure type."""
import json
from collections import Counter

from datasets import load_dataset
from tqdm import tqdm

ds = load_dataset("nvidia/Open-SWE-Traces", "v1.0", split="openhands", streaming=True)

shown = 0
endings = Counter()
patch_lengths = {0: [], 1: []}
for ex in tqdm(ds.take(600), total=600, desc="inspect", unit="traj"):
    if ex["resolved"] not in (0, 1):
        continue
    metadata = ex.get("metadata") or {}
    model_patch = metadata.get("model_patch")
    patch = model_patch.get("patch") if isinstance(model_patch, dict) else model_patch
    patch_lengths[ex["resolved"]].append(len(patch.strip()) if isinstance(patch, str) else -1)

    last_assistant = next((m for m in reversed(ex["messages"]) if m.get("role") == "assistant"), {})
    calls = last_assistant.get("tool_calls") or []
    last_tool = calls[-1]["function"]["name"] if calls else "no_tool_call"
    called_finish = any(
        (c.get("function") or {}).get("name") == "finish"
        for m in ex["messages"] if m.get("role") == "assistant"
        for c in (m.get("tool_calls") or [])
    )
    endings[(ex["resolved"], last_tool, called_finish)] += 1

    if ex["resolved"] == 0 and shown < 2:
        shown += 1
        print("\n--- failed example ---")
        print("model_patch type:", type(model_patch).__name__)
        if isinstance(model_patch, dict):
            print("model_patch keys:", list(model_patch.keys()))
            print("num_modified_lines:", model_patch.get("num_modified_lines"))
        print("patch start:", json.dumps(patch)[:300] if patch is not None else None)
        print("last message role:", ex["messages"][-1].get("role"), "| last assistant tool:", last_tool)

print("\nPatch length (chars) by resolved: min / median / max")
for r, lengths in patch_lengths.items():
    if lengths:
        s = sorted(lengths)
        print(f"  resolved={r}: n={len(s)}  min={s[0]}  median={s[len(s) // 2]}  max={s[-1]}  zero_count={sum(1 for x in s if x == 0)}")

print("\nEndings: (resolved, last assistant tool, ever called finish) -> count")
for key, count in sorted(endings.items(), key=lambda kv: -kv[1]):
    print(f"  {key}: {count}")