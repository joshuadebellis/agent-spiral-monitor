"""Matched-false-alarm comparison of rolling detectors on hit_limit runs: topology vs repetition vs both.
Reads context_per_turn.csv written by context_check.py (needs the 'run' column)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

FA_LEVELS = [0.05, 0.10, 0.20]
EARLY = 10
N_BOOT = 500
DETECTORS = ["topo", "rep", "both"]
NAMES = {"topo": "topology", "rep": "repetition", "both": "topology + repetition"}


def prepare(pt, seed):
    pt = pt[pt["topo"].notna() & pt["rep"].notna()].copy()
    rng = np.random.default_rng(seed)
    pt["rep"] = pt["rep"] + rng.uniform(0.0, 1e-6, len(pt))
    succ = pt[pt["group"] == "success"]
    for c in ["topo", "rep"]:
        ref = np.sort(succ[c].to_numpy())
        pt[f"{c}_pct"] = np.searchsorted(ref, pt[c].to_numpy(), side="right") / len(ref)
    pt["both"] = (pt["topo_pct"] + pt["rep_pct"]) / 2.0
    return pt


def run_table(pt):
    g = pt.groupby("run")
    rt = pd.DataFrame({"group": g["group"].first()})
    early = pt[pt["turns_before_end"] >= EARLY].groupby("run")
    for d in DETECTORS:
        rt[f"{d}_max"] = g[d].max()
        rt[f"{d}_early"] = early[d].max()
        rt[f"{d}_early"] = rt[f"{d}_early"].fillna(-np.inf)
    return rt


def detection(succ_max, hit_max, hit_early, fa):
    thr = np.quantile(succ_max, 1.0 - fa)
    return thr, float((succ_max > thr).mean()), float((hit_max > thr).mean()), float((hit_early > thr).mean())


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config_nebius_confirm.yaml"
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    seed = cfg["seed"]
    out_dir = Path(cfg["results_dir"]) / f"anomaly_{cfg['tag']}"
    pt = pd.read_csv(out_dir / "context_per_turn.csv")
    if "run" not in pt.columns:
        raise SystemExit("context_per_turn.csv has no 'run' column: update context_check.py and rerun it first.")
    pt = prepare(pt, seed)
    rt = run_table(pt)
    succ = rt[rt["group"] == "success"]
    hit = rt[rt["group"] == "hit_limit"]
    print(f"{len(succ)} success runs, {len(hit)} hit_limit runs (held-out tasks). "
          f"Early = alarm fires at least {EARLY} turns before the cutoff.")

    win_succ = pt[pt["group"] == "success"]
    win_hit_early = pt[(pt["group"] == "hit_limit") & (pt["turns_before_end"] >= EARLY)]
    y_win = np.r_[np.zeros(len(win_succ)), np.ones(len(win_hit_early))]
    print("\n=== Window-level AUC: hit_limit windows >= 10 turns before cutoff vs all success windows ===")
    for d in DETECTORS:
        auc = roc_auc_score(y_win, np.r_[win_succ[d].to_numpy(), win_hit_early[d].to_numpy()])
        print(f"  {NAMES[d]:<24} {auc:.3f}")

    rows = []
    for d in DETECTORS:
        for fa in FA_LEVELS:
            thr, fa_ach, det_any, det_early = detection(
                succ[f"{d}_max"].to_numpy(), hit[f"{d}_max"].to_numpy(), hit[f"{d}_early"].to_numpy(), fa)
            fired = pt[(pt["group"] == "hit_limit") & (pt[d] > thr)]
            lead = fired.groupby("run")["turns_before_end"].max()
            ctx = fired.sort_values("turns_before_end", ascending=False).groupby("run")["ctx_frac"].first()
            rows.append({
                "detector": NAMES[d], "fa_target": fa, "fa_achieved": fa_ach,
                "detect_any": det_any, "detect_early": det_early,
                "median_lead_turns": float(lead.median()) if len(lead) else np.nan,
                "median_ctx_at_alarm": float(ctx.median()) if len(ctx) else np.nan,
            })
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "alarm_curve.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== Detection of hit_limit runs at matched false-alarm rates on successful runs ===")
    print(table.round(3).to_string(index=False))

    rng = np.random.default_rng(seed)
    s_idx = np.arange(len(succ))
    h_idx = np.arange(len(hit))
    arrays = {d: (succ[f"{d}_max"].to_numpy(), hit[f"{d}_max"].to_numpy(), hit[f"{d}_early"].to_numpy())
              for d in DETECTORS}
    diffs = {(c, fa): [] for c in ["topo", "both"] for fa in FA_LEVELS}
    for _ in tqdm(range(N_BOOT), desc="bootstrap over runs", unit="boot"):
        sb = rng.choice(s_idx, len(s_idx))
        hb = rng.choice(h_idx, len(h_idx))
        for fa in FA_LEVELS:
            base = detection(arrays["rep"][0][sb], arrays["rep"][1][hb], arrays["rep"][2][hb], fa)[3]
            for c in ["topo", "both"]:
                val = detection(arrays[c][0][sb], arrays[c][1][hb], arrays[c][2][hb], fa)[3]
                diffs[(c, fa)].append(val - base)

    out = []
    for (c, fa), v in diffs.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        out.append({"comparison": f"{NAMES[c]} minus repetition", "fa_target": fa,
                    "early_detect_diff": float(np.mean(v)), "lo": float(lo), "hi": float(hi),
                    "verdict": "better" if lo > 0 else ("worse" if hi < 0 else "no difference")})
    dtab = pd.DataFrame(out)
    dtab.to_csv(out_dir / "alarm_curve_diffs.csv", index=False)
    print("\n=== Early detection (>= 10 turns ahead): difference vs repetition alone, 95% bootstrap CI over runs ===")
    print(dtab.round(3).to_string(index=False))
    print(f"\nWrote {out_dir / 'alarm_curve.csv'} and {out_dir / 'alarm_curve_diffs.csv'}")


if __name__ == "__main__":
    main()