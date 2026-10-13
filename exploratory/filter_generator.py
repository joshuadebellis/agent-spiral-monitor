"""Write a copy of a feature set restricted to one generator model."""
import pickle
import sys
from pathlib import Path

import pandas as pd

source_tag = sys.argv[1]
generator = sys.argv[2]
new_tag = sys.argv[3]
data_dir = Path("data")

df = pd.read_parquet(data_dir / f"features_{source_tag}.parquet")
sub = df[df["generator"] == generator].reset_index(drop=True)
sub.to_parquet(data_dir / f"features_{new_tag}.parquet", index=False)

with open(data_dir / f"traces_{source_tag}.pkl", "rb") as f:
    traces = pickle.load(f)
with open(data_dir / f"traces_{new_tag}.pkl", "wb") as f:
    pickle.dump({k: v for k, v in traces.items() if k in set(sub["id"])}, f)

print(f"{generator}: {len(sub)} trajectories, failure rate {(~sub['final_answer_correct']).mean():.3f}")
print(sub.groupby("split").size().sort_values(ascending=False).to_string())