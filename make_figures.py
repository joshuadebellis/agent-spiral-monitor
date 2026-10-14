"""Figure for the spiral-monitor write-up. Reads saved confirm-run results only; recomputes nothing expensive."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

EARLY = 10
MAX_TBE = 40
MIN_RUNS = 30
FAIL = "#D55E00"
OK = "#0072B2"
GROUPS = [("hit_limit", FAIL, "Runs that hit the context limit"), ("success", OK, "Successful runs")]
DETECTORS = [
    ("topo_starts", "Topology, label-free", "#6A3D9A", "-"),
    ("topo_success", "Topology, success-referenced", "#6A3D9A", "--"),
    ("rep", "Repeat counter (exact string match)", "#7F7F7F", "-"),
]


def midrank_vs_success(pt, col):
    ref = np.sort(pt.loc[pt["group"] == "success", col].to_numpy())
    v = pt[col].to_numpy()
    lo = np.searchsorted(ref, v, side="left")
    hi = np.searchsorted(ref, v, side="right")
    return (lo + hi) / 2.0 / len(ref)


def early_detection_curve(pt, col, fas, seed):
    rng = np.random.default_rng(seed)
    tmp = pt[["run", "group", "turns_before_end"]].copy()
    tmp["s"] = pt[col].to_numpy() + rng.uniform(0.0, 1e-7, len(pt))
    runs = tmp.groupby("run").agg(group=("group", "first"), smax=("s", "max"))
    early = tmp[tmp["turns_before_end"] >= EARLY].groupby("run")["s"].max()
    runs["early"] = early.reindex(runs.index).fillna(-np.inf)
    succ = runs.loc[runs["group"] == "success", "smax"].to_numpy()
    hit = runs.loc[runs["group"] == "hit_limit", "early"].to_numpy()
    return np.array([(hit > np.quantile(succ, 1.0 - fa)).mean() for fa in fas])


def plot_profile(ax, pt, col, title):
    sub = pt[pt["turns_before_end"] <= MAX_TBE]
    for group, color, label in GROUPS:
        g = sub[sub["group"] == group].groupby("turns_before_end")[col]
        stats = pd.DataFrame({"mean": g.mean(), "std": g.std(), "n": g.count()})
        stats = stats[stats["n"] >= MIN_RUNS]
        half = 1.96 * stats["std"] / np.sqrt(stats["n"])
        ax.plot(stats.index, stats["mean"].values, color=color, lw=2, label=label)
        ax.fill_between(stats.index, (stats["mean"] - half).values, (stats["mean"] + half).values,
                        color=color, alpha=0.18, lw=0)
    ax.axvline(EARLY, color="#999999", ls=":", lw=1)
    ax.axhline(0.5, color="#cccccc", lw=0.8)
    ax.text(MAX_TBE - 0.5, 0.515, "typical successful run", fontsize=8, color="#888888", va="bottom")
    ax.set_xlim(MAX_TBE, 0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Turns before the run ends")
    ax.set_ylabel("Score percentile vs. successful runs")
    ax.set_title(title, loc="left", fontsize=11)


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config_nebius_confirm.yaml"
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    tag = cfg["tag"]
    seed = cfg["seed"]
    res_dir = Path(cfg["results_dir"]) / f"anomaly_{tag}"
    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    pt = pd.read_csv(res_dir / "label_free_per_turn.csv")
    for col in ["topo_starts", "rep"]:
        pt[f"{col}_pct"] = midrank_vs_success(pt, col)

    fas = np.round(np.linspace(0.01, 0.50, 50), 2)
    curves = {col: early_detection_curve(pt, col, fas, seed) for col, _, _, _ in DETECTORS}

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    plot_profile(ax_a, pt, "topo_starts_pct", "A. Label-free topology score through the run")
    ax_a.legend(frameon=False, loc="upper left")
    plot_profile(ax_b, pt, "rep_pct", "B. Repeat counter (exact string match) through the run")

    for col, label, color, ls in DETECTORS:
        ax_c.plot(fas, curves[col], color=color, ls=ls, lw=2, label=label)
    ax_c.axvline(0.10, color="#999999", ls=":", lw=1)
    ax_c.set_xlim(0, 0.5)
    ax_c.set_ylim(0, 1)
    ax_c.set_xlabel("False alarms: share of successful runs flagged")
    ax_c.set_ylabel(f"Failing runs flagged at least {EARLY} turns early")
    ax_c.set_title("C. Early warning at matched false-alarm rates", loc="left", fontsize=11)
    ax_c.legend(frameon=False, loc="lower right")

    fig.suptitle("Agents that run out of context spiral first, and a label-free topological monitor sees it coming",
                 fontsize=13, x=0.01, ha="left")
    png = fig_dir / "spiral_monitor.png"
    svg = fig_dir / "spiral_monitor.svg"
    fig.savefig(png, dpi=200)
    fig.savefig(svg)

    i10 = int(np.argmin(np.abs(fas - 0.10)))
    print("\nEarly detection at 10% false alarms:")
    for col, label, _, _ in DETECTORS:
        print(f"  {label:<38} {curves[col][i10]:.3f}")
    print(f"\nWrote {png} and {svg}")


if __name__ == "__main__":
    main()