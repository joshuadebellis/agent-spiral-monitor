"""Print the last few actions of some success and hit_limit runs, to check for end-of-run artifacts."""
from datasets import load_dataset

from build_nebius_anomaly import outcome_group, turns_from_trajectory

cfg = {"max_reasoning_chars": 200, "max_action_chars": 150, "max_observation_chars": 150}
ds = load_dataset("nebius/SWE-agent-trajectories", split="train", streaming=True)
shown = {"success": 0, "hit_limit": 0}
for ex in ds:
    if ex["model_name"] != "swe-agent-llama-70b":
        continue
    group = outcome_group(bool(ex["target"]), ex["exit_status"])
    if group not in shown or shown[group] >= 4:
        continue
    turns = turns_from_trajectory(ex["trajectory"], cfg)
    if len(turns) < 15:
        continue
    shown[group] += 1
    print(f"\n===== {group} | {ex['exit_status']} | {len(turns)} turns =====")
    for t in turns[-5:]:
        print("  ACTION:", repr(t["action"][:150]))
        print("     OBS:", repr(t["observation"][:100]))
    if all(v >= 4 for v in shown.values()):
        break