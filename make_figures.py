"""Figure for the spiral-monitor write-up. Reads saved confirm-run results only; recomputes nothing expensive."""
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

W = 15
EARLY = 10
MAX_TBE = 40
MIN_RUNS = 30
BAND = (31, 69)
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


def bar_lengths(diagrams, ids):
    rows = []
    for i in tqdm(list(ids), desc="H0 bars, last 15 actions", unit="run"):
        d = np.asarray(diagrams[i]["last"]["action"]["rips_h0"], dtype=float).reshape(-1, 2)
        deaths = d[np.isfinite(d[:, 1]), 1]
        bars = np.sort(deaths)[::-1][: W - 1]
        rows.append(np.pad(bars, (0, W - 1 - len(bars))))
    return np.vstack(rows)


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
    data_dir = Path(cfg["data_dir"])
    res_dir = Path(cfg["results_dir"]) / f"anomaly_{tag}"
    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    pt = pd.read_csv(res_dir / "label_free_per_turn.csv")
    for col in ["topo_starts", "rep"]:
        pt[f"{col}_pct"] = midrank_vs_success(pt, col)

    df = pd.read_parquet(data_dir / f"anomaly_{tag}.parquet")
    df = df[(df["n_turns"] >= BAND[0]) & (df["n_turns"] <= BAND[1]) & df["group"].isin(["success", "hit_limit"])]
    with open(data_dir / f"anomaly_{tag}_diagrams.pkl", "rb") as f:
        diagrams = pickle.load(f)
    bars = {g: bar_lengths(diagrams, df.loc[df["group"] == g, "id"]) for g, _, _ in GROUPS}

    fas = np.round(np.linspace(0.01, 0.50, 50), 2)
    curves = {col: early_detection_curve(pt, col, fas, seed) for col, _, _, _ in DETECTORS}

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax = axes[0, 0]
    ranks = np.arange(1, W)
    for g, color, label in GROUPS:
        b = bars[g]
        ax.plot(ranks, b.mean(axis=0), color=color, lw=2, marker="o", ms=4, label=f"{label} (n={len(b)})")
        ax.fill_between(ranks, np.percentile(b, 25, axis=0), np.percentile(b, 75, axis=0),
                        color=color, alpha=0.18, lw=0)
    ax.set_xticks(ranks)
    ax.set_xlabel("H0 bar, longest to shortest")
    ax.set_ylabel("Bar length (cosine distance)")
    ax.set_title(f"A. Shape of the last 15 actions (runs of {BAND[0]}-{BAND[1]} turns)", loc="left", fontsize=11)
    ax.text(0.98, 0.55, "Bars near zero = steps that are\nnear-copies of another step",
            transform=ax.transAxes, ha="right", va="center", fontsize=9, color="#555555")
    ax.legend(frameon=False, loc="upper right")

    plot_profile(axes[0, 1], pt, "topo_starts_pct", "B. Label-free topology score through the run")
    axes[0, 1].legend(frameon=False, loc="upper left")
    plot_profile(axes[1, 0], pt, "rep_pct", "C. Repeat counter (exact string match) through the run")

    ax = axes[1, 1]
    for col, label, color, ls in DETECTORS:
        ax.plot(fas, curves[col], color=color, ls=ls, lw=2, label=label)
    ax.axvline(0.10, color="#999999", ls=":", lw=1)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False alarms: share of successful runs flagged")
    ax.set_ylabel(f"Failing runs flagged at least {EARLY} turns early")
    ax.set_title("D. Early warning at matched false-alarm rates", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower right")

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
    print("\nMean H0 bar lengths, last 15 actions (longest to shortest):")
    for g, _, label in GROUPS:
        print(f"  {label:<32} " + " ".join(f"{v:.3f}" for v in bars[g].mean(axis=0)))
    print(f"\nWrote {png} and {svg}")


if __name__ == "__main__":
    main()