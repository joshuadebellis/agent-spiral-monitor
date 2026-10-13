"""Label-free variants of the rolling late-collapse monitor, compared at matched false-alarm rates.
Detectors (scored on held-out tasks; successful runs are used ONLY to set and measure the false-alarm rate):
  topo_success : kNN distance to rolling windows from known-successful runs (one-class: needs success labels)
  topo_starts  : kNN distance to the first 15 turns of ANY run on reference tasks (no outcome labels at all)
  topo_self    : distance from the run's own first 15 turns (no reference set at all)
  rep          : rolling repeat rate (hand-built after looking at failures)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ripser import ripser
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from anomaly_eval import fit_imager, normalize_rows
from context_check import ROLL, W, dup_flags, images_with_progress, rolling_mean, stream_rows
from trace_features import cosine_distance_matrix

FA_LEVELS = [0.05, 0.10, 0.20]
EARLY = 10
N_BOOT = 500
DETECTORS = ["topo_success", "topo_starts", "topo_self", "rep"]
PAIRS = [
    ("topo_success", "rep"), ("topo_starts", "rep"), ("topo_self", "rep"),
    ("topo_starts", "topo_success"), ("topo_self", "topo_success"),
]
TBE_BINS = [-1, 4, 9, 19, 29, 10_000]
TBE_LABELS = ["0-4", "5-9", "10-19", "20-29", "30+"]


def knn_score(ref, query, k):
    nn = NearestNeighbors(n_neighbors=min(k, len(ref))).fit(ref)
    dist, _ = nn.kneighbors(query)
    return dist.mean(axis=1)


def detection(succ_max, hit_max, hit_early, fa):
    thr = np.quantile(succ_max, 1.0 - fa)
    return thr, float((succ_max > thr).mean()), float((hit_max > thr).mean()), float((hit_early > thr).mean())


def analyze(pt, seed, out_dir):
    pt = pt.copy()
    rng = np.random.default_rng(seed)
    for d in DETECTORS:
        pt[d] = pt[d] + rng.uniform(0.0, 1e-7, len(pt))

    g = pt.groupby("run")
    rt = pd.DataFrame({"group": g["group"].first()})
    early = pt[pt["turns_before_end"] >= EARLY].groupby("run")
    for d in DETECTORS:
        rt[f"{d}_max"] = g[d].max()
        rt[f"{d}_early"] = early[d].max()
        rt[f"{d}_early"] = rt[f"{d}_early"].fillna(-np.inf)
    succ = rt[rt["group"] == "success"]
    hit = rt[rt["group"] == "hit_limit"]
    print(f"\n{len(succ)} success runs, {len(hit)} hit_limit runs on held-out tasks. "
          f"Early = alarm at least {EARLY} turns before the cutoff.")

    win_succ = pt[pt["group"] == "success"]
    win_hit = pt[(pt["group"] == "hit_limit") & (pt["turns_before_end"] >= EARLY)]
    y_win = np.r_[np.zeros(len(win_succ)), np.ones(len(win_hit))]
    print("\n=== Window-level AUC: hit_limit windows >= 10 turns before cutoff vs all success windows ===")
    for d in DETECTORS:
        print(f"  {d:<13} {roc_auc_score(y_win, np.r_[win_succ[d].to_numpy(), win_hit[d].to_numpy()]):.3f}")

    rows = []
    for d in DETECTORS:
        for fa in FA_LEVELS:
            thr, fa_ach, det_any, det_early = detection(
                succ[f"{d}_max"].to_numpy(), hit[f"{d}_max"].to_numpy(), hit[f"{d}_early"].to_numpy(), fa)
            fired = pt[(pt["group"] == "hit_limit") & (pt[d] > thr)]
            lead = fired.groupby("run")["turns_before_end"].max()
            rows.append({"detector": d, "fa_target": fa, "fa_achieved": fa_ach, "detect_any": det_any,
                         "detect_early": det_early,
                         "median_lead_turns": float(lead.median()) if len(lead) else np.nan})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "label_free_detection.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== Detection of hit_limit runs at matched false-alarm rates ===")
    print(table.round(3).to_string(index=False))

    arrays = {d: (succ[f"{d}_max"].to_numpy(), hit[f"{d}_max"].to_numpy(), hit[f"{d}_early"].to_numpy())
              for d in DETECTORS}
    diffs = {(a, b, fa): [] for a, b in PAIRS for fa in FA_LEVELS}
    for _ in tqdm(range(N_BOOT), desc="bootstrap over runs", unit="boot"):
        sb = rng.choice(len(succ), len(succ))
        hb = rng.choice(len(hit), len(hit))
        for fa in FA_LEVELS:
            vals = {d: detection(arrays[d][0][sb], arrays[d][1][hb], arrays[d][2][hb], fa)[3] for d in DETECTORS}
            for a, b in PAIRS:
                diffs[(a, b, fa)].append(vals[a] - vals[b])
    out = []
    for (a, b, fa), v in diffs.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        out.append({"comparison": f"{a} minus {b}", "fa_target": fa, "early_detect_diff": float(np.mean(v)),
                    "lo": float(lo), "hi": float(hi),
                    "verdict": "better" if lo > 0 else ("worse" if hi < 0 else "no difference")})
    dtab = pd.DataFrame(out)
    dtab.to_csv(out_dir / "label_free_diffs.csv", index=False)
    print("\n=== Early detection (>= 10 turns ahead): pairwise differences, 95% bootstrap CI over runs ===")
    print(dtab.round(3).to_string(index=False))

    pt["tbe_bin"] = pd.cut(pt["turns_before_end"], TBE_BINS, labels=TBE_LABELS)
    print("\nMean score by turns before the end of the run:")
    print(pt.pivot_table(index="tbe_bin", columns="group", values=DETECTORS, aggfunc="mean", observed=False)
          .round(3).to_string())
    print(f"\nWrote label_free_detection.csv and label_free_diffs.csv to {out_dir}")


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config_nebius_confirm.yaml"
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    tag = cfg["tag"]
    seed = cfg["seed"]
    pixels = int(cfg["image_pixels"])
    k = int(cfg["knn_k"])
    data_dir = Path(cfg["data_dir"])
    out_dir = Path(cfg["results_dir"]) / f"anomaly_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_dir / f"anomaly_{tag}.parquet")
    streamed = stream_rows(cfg, df)
    df = df[df["id"].isin(streamed)].reset_index(drop=True)
    grp = df["group"].to_numpy()
    inst = df["instance_id"].to_numpy()
    print(f"\nMatched {len(df)} rows (all groups, including wrong, feed the label-free reference).")

    texts = []
    spans = []
    for i in df["id"]:
        a = streamed[i]["actions"]
        spans.append((len(texts), len(texts) + len(a)))
        texts.extend(a)
    model = SentenceTransformer(cfg["embedding_model"], device=cfg["device"])
    X = model.encode(texts, batch_size=512, normalize_embeddings=cfg["normalize_embeddings"],
                     show_progress_bar=True, convert_to_numpy=True)

    win_dgms = []
    win_run = []
    win_t = []
    for r, (a, b) in enumerate(tqdm(spans, desc="rolling 15-turn diagrams", unit="run")):
        Xr = X[a:b]
        for t in range(W - 1, b - a):
            dg = ripser(cosine_distance_matrix(Xr[t - W + 1: t + 1]), distance_matrix=True, maxdim=0)["dgms"][0]
            dg = np.asarray(dg, dtype=np.float32).reshape(-1, 2)
            win_dgms.append(dg[np.isfinite(dg[:, 1])])
            win_run.append(r)
            win_t.append(t)
    win_run = np.array(win_run)
    win_t = np.array(win_t)
    imgs = normalize_rows(images_with_progress(win_dgms, fit_imager(win_dgms, pixels), pixels, "rolling images"))

    rng = np.random.default_rng(seed)
    uniq = np.unique(inst)
    ref_inst = set(rng.choice(uniq, size=len(uniq) // 2, replace=False))
    is_ref = np.array([i in ref_inst for i in inst])
    eval_runs = np.where(~is_ref & np.isin(grp, ["success", "hit_limit"]))[0]
    ref_success_runs = np.where(is_ref & (grp == "success"))[0]
    ref_any_runs = np.where(is_ref)[0]

    eval_mask = np.isin(win_run, eval_runs)
    ref_success_mask = np.isin(win_run, ref_success_runs)
    ref_starts_mask = np.isin(win_run, ref_any_runs) & (win_t == W - 1)
    print(f"Reference: {int(ref_success_mask.sum())} success windows; {int(ref_starts_mask.sum())} run starts "
          f"(all outcomes). Evaluating {len(eval_runs)} runs on held-out tasks.")

    q = imgs[eval_mask]
    s_success = knn_score(imgs[ref_success_mask], q, k)
    s_starts = knn_score(imgs[ref_starts_mask], q, k)
    first_row = {int(r): int(i) for i, (r, t) in enumerate(zip(win_run, win_t)) if t == W - 1}
    q_runs = win_run[eval_mask]
    own_start = imgs[[first_row[int(r)] for r in q_runs]]
    s_self = np.linalg.norm(q - own_start, axis=1)

    reps = {int(r): rolling_mean(dup_flags(streamed[df["id"][r]]["actions"]), ROLL) for r in eval_runs}
    rows = []
    for r, t, a1, a2, a3 in zip(q_runs, win_t[eval_mask], s_success, s_starts, s_self):
        r = int(r)
        n = spans[r][1] - spans[r][0]
        rows.append({
            "run": r, "group": grp[r], "turns_before_end": n - 1 - int(t),
            "ctx_frac": streamed[df["id"][r]]["cum"][t] / streamed[df["id"][r]]["total"],
            "topo_success": a1, "topo_starts": a2, "topo_self": a3, "rep": reps[r][t],
        })
    pt = pd.DataFrame(rows)
    pt.to_csv(out_dir / "label_free_per_turn.csv", index=False)
    analyze(pt, seed, out_dir)


if __name__ == "__main__":
    main()