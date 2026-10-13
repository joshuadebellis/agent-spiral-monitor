"""Context-limit checks on a built set (default: confirm).
1) Does P1 (late collapse) still add beyond length + repetition + total characters (a token-size proxy)?
2) When do action loops start in hit_limit runs: turns before the end, and fraction of context used?
3) Rolling late-collapse score along each run: how far ahead of the cutoff does it fire?"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset
from ripser import ripser
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from add_repetition import REP_FEATS, norm_text
from anomaly_eval import anomaly_scores, boot_auc_and_delta, combined_scores, fit_imager, normalize_rows
from build_nebius_anomaly import POINT_TYPES, turns_from_trajectory
from trace_features import cosine_distance_matrix

W = 15
DUP_LOOKBACK = 5
ROLL = 6
FALSE_ALARM = 0.10
TBE_BINS = [-1, 4, 9, 19, 29, 10_000]
TBE_LABELS = ["0-4", "5-9", "10-19", "20-29", "30+"]
CTX_BINS = [0.0, 0.25, 0.5, 0.75, 1.0001]
CTX_LABELS = ["0-25%", "25-50%", "50-75%", "75-100%"]


def turn_context(trajectory):
    cum = 0
    out = []
    for step in trajectory:
        cum += len(step.get("text") or "")
        if step.get("role") == "ai":
            out.append(cum)
    return np.array(out, dtype=float), float(cum)


def dup_flags(actions):
    a = [norm_text(x) for x in actions]
    return np.array([float(a[i] in a[max(0, i - DUP_LOOKBACK): i]) for i in range(len(a))])


def rolling_mean(x, k):
    c = np.cumsum(np.r_[0.0, x])
    out = np.full(len(x), np.nan)
    for i in range(k - 1, len(x)):
        out[i] = (c[i + 1] - c[i + 1 - k]) / k
    return out


def stream_rows(cfg, df):
    wanted = {
        int(row_id.rsplit("#", 1)[1]): (iid, int(n))
        for row_id, iid, n in zip(df["id"], df["instance_id"], df["n_turns"])
    }
    last_pos = max(wanted)
    ds = load_dataset(cfg["dataset"], split=cfg["split"], streaming=True)
    if cfg.get("stream_skip"):
        ds = ds.skip(int(cfg["stream_skip"]))
    if cfg.get("shuffle_buffer"):
        ds = ds.shuffle(seed=cfg["seed"], buffer_size=int(cfg["shuffle_buffer"]))
    out = {}
    scanned = 0
    with tqdm(total=len(wanted), desc=f"re-streaming {cfg['tag']}", unit="traj") as bar:
        for ex in ds:
            scanned += 1
            if scanned > last_pos:
                break
            if scanned not in wanted:
                continue
            bar.update(1)
            iid, n_expected = wanted[scanned]
            turns = turns_from_trajectory(ex["trajectory"], cfg)
            if ex["instance_id"] != iid or len(turns) != n_expected:
                continue
            cum, total = turn_context(ex["trajectory"])
            out[f"{iid}#{scanned}"] = {"actions": [t["action"] for t in turns], "cum": cum, "total": total}
    return out


def images_with_progress(dgms, imager, pixels, desc):
    out = np.zeros((len(dgms), pixels * pixels), dtype=np.float64)
    if imager is None:
        return out
    for i, d in enumerate(tqdm(dgms, desc=desc, unit="dgm")):
        if len(d):
            img = np.asarray(imager.transform(np.asarray(d, dtype=np.float64), skew=True), dtype=np.float64)
            flat = img.ravel()
            out[i, : flat.size] = flat[: pixels * pixels]
    return out


def p1_with_chars(df, diagrams, cfg, band):
    seed = cfg["seed"]
    folds = cfg["cv_folds"]
    pixels = int(cfg["image_pixels"])
    n_boot = int(cfg["n_bootstrap"])
    lo, hi = band
    d = df[(df["n_turns"] >= lo) & (df["n_turns"] <= hi)].reset_index(drop=True)
    failed = d["failed"].to_numpy()
    turns = d["n_turns"].to_numpy(dtype=float)
    groups = d["instance_id"].to_numpy()

    dgms = [diagrams[i]["last"]["action"]["rips_h0"] for i in d["id"]]
    images = normalize_rows(images_with_progress(dgms, fit_imager(dgms, pixels), pixels, f"P1 images {lo}-{hi}"))
    s_all = anomaly_scores(images, failed, turns, groups, folds, seed, int(cfg["knn_k"]))["knn"]

    m = d["group"].isin(["success", "hit_limit"]).to_numpy()
    y = failed[m]
    g = groups[m]
    s = s_all[m]
    total = d["total_chars"].to_numpy(dtype=float)[m]
    last = d["last_chars"].to_numpy(dtype=float)[m]
    L = np.log1p(turns[m]).reshape(-1, 1)
    R = d.loc[m, [f"last_{p}_{f}" for p in POINT_TYPES for f in REP_FEATS]].to_numpy(dtype=float)
    C = np.c_[np.log1p(total), np.log1p(total / turns[m]), np.log1p(last)]

    base_lc = combined_scores(np.c_[L, C], y, g, folds, seed)
    base_lr = combined_scores(np.c_[L, R], y, g, folds, seed)
    full_lr = combined_scores(np.c_[L, R, s], y, g, folds, seed)
    base_lrc = combined_scores(np.c_[L, R, C], y, g, folds, seed)
    full_lrc = combined_scores(np.c_[L, R, C, s], y, g, folds, seed)
    _, _, d1_lo, d1_hi = boot_auc_and_delta(y, s, base_lr, full_lr, n_boot, seed)
    _, _, d2_lo, d2_hi = boot_auc_and_delta(y, s, base_lrc, full_lrc, n_boot, seed)
    return {
        "band": f"{lo}-{hi}" if hi < 100000 else f"{lo}+",
        "n": int(m.sum()),
        "n_success": int((y == 0).sum()),
        "auc_topo": float(roc_auc_score(y, s)),
        "auc_total_chars": float(roc_auc_score(y, total)),
        "auc_len_chars": float(roc_auc_score(y, base_lc)),
        "auc_len_rep": float(roc_auc_score(y, base_lr)),
        "auc_len_rep_chars": float(roc_auc_score(y, base_lrc)),
        "gain_vs_len_rep": float(roc_auc_score(y, full_lr) - roc_auc_score(y, base_lr)),
        "gain_vs_len_rep_lo": d1_lo,
        "gain_vs_len_rep_hi": d1_hi,
        "gain_vs_len_rep_chars": float(roc_auc_score(y, full_lrc) - roc_auc_score(y, base_lrc)),
        "gain_vs_len_rep_chars_lo": d2_lo,
        "gain_vs_len_rep_chars_hi": d2_hi,
        "verdict": "PASS" if d2_lo > 0 else "fail",
    }


def alarm_onsets(series, runs, grp, cum, total, label):
    succ = [r for r in runs if grp[r] == "success"]
    hit = [r for r in runs if grp[r] == "hit_limit"]
    smax = np.array([np.nanmax(series[r]) for r in succ])
    thr = float(np.quantile(smax, 1 - FALSE_ALARM))
    fa_rate = float((smax > thr).mean())
    rows = []
    for r in hit:
        s = series[r]
        idx = np.where(np.nan_to_num(s, nan=-np.inf) > thr)[0]
        if len(idx) == 0:
            continue
        t = int(idx[0])
        n = len(s)
        rows.append({
            "detector": label,
            "turns_before_end": n - 1 - t,
            "turn_frac": t / (n - 1),
            "ctx_frac": cum[r][t] / total[r],
        })
    onsets = pd.DataFrame(rows)
    summary = {
        "detector": label,
        "threshold": thr,
        "false_alarm_success_runs": fa_rate,
        "detected_hit_limit_runs": len(rows) / len(hit),
        "median_turns_before_end": float(onsets["turns_before_end"].median()) if len(rows) else np.nan,
        "median_ctx_frac_at_onset": float(onsets["ctx_frac"].median()) if len(rows) else np.nan,
        "share_onset_before_half_context": float((onsets["ctx_frac"] < 0.5).mean()) if len(rows) else np.nan,
        "share_onset_10plus_turns_early": float((onsets["turns_before_end"] >= 10).mean()) if len(rows) else np.nan,
    }
    return summary, onsets


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
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    df = pd.read_parquet(data_dir / f"anomaly_{tag}.parquet")
    rep = pd.read_parquet(data_dir / f"repetition_{tag}.parquet")
    df = df.merge(rep, on="id", how="inner").reset_index(drop=True)
    with open(data_dir / f"anomaly_{tag}_diagrams.pkl", "rb") as f:
        diagrams = pickle.load(f)

    streamed = stream_rows(cfg, df)
    df = df[df["id"].isin(streamed)].reset_index(drop=True)
    print(f"\nMatched {len(df)} rows.")
    df["total_chars"] = [streamed[i]["total"] for i in df["id"]]
    df["last_chars"] = [
        streamed[i]["total"] - streamed[i]["cum"][len(streamed[i]["cum"]) - W - 1]
        if len(streamed[i]["cum"]) > W else streamed[i]["total"]
        for i in df["id"]
    ]
    print("\nTotal characters per run (token-size proxy), by group:")
    print(df.groupby("group")["total_chars"].describe()[["count", "25%", "50%", "75%"]].round(0).to_string())

    # ---- 1) P1 with a character baseline
    p1 = pd.DataFrame([p1_with_chars(df, diagrams, cfg, b) for b in [(31, 69), (15, 100000)]])
    p1.to_csv(out_dir / "context_p1_chars.csv", index=False)
    print("\n=== 1) P1 late collapse vs a token-size baseline ===")
    print(p1[["band", "n", "n_success", "auc_topo", "auc_total_chars", "auc_len_chars",
              "auc_len_rep", "auc_len_rep_chars"]].round(3).to_string(index=False))
    print(p1[["band", "gain_vs_len_rep", "gain_vs_len_rep_lo", "gain_vs_len_rep_hi",
              "gain_vs_len_rep_chars", "gain_vs_len_rep_chars_lo", "gain_vs_len_rep_chars_hi",
              "verdict"]].round(3).to_string(index=False))

    # ---- 2) and 3) rolling signals along each run
    sub = df[df["group"].isin(["success", "hit_limit"])].reset_index(drop=True)
    grp = sub["group"].to_numpy()
    cum = [streamed[i]["cum"] for i in sub["id"]]
    total = [streamed[i]["total"] for i in sub["id"]]
    texts = []
    spans = []
    for i in sub["id"]:
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
    inst = sub["instance_id"].to_numpy()
    uniq = np.unique(inst)
    ref_inst = set(rng.choice(uniq, size=len(uniq) // 2, replace=False))
    is_ref = np.array([i in ref_inst for i in inst])
    ref_runs = np.where(is_ref & (grp == "success"))[0]
    eval_runs = np.where(~is_ref)[0]
    print(f"\nRolling detector: {len(ref_runs)} reference success runs; evaluating "
          f"{int((grp[eval_runs] == 'success').sum())} success and {int((grp[eval_runs] == 'hit_limit').sum())} hit_limit runs "
          f"on held-out tasks.")

    ref_mask = np.isin(win_run, ref_runs)
    eval_mask = np.isin(win_run, eval_runs)
    nn = NearestNeighbors(n_neighbors=min(k, int(ref_mask.sum()))).fit(imgs[ref_mask])
    dist, _ = nn.kneighbors(imgs[eval_mask])
    topo = {r: np.full(spans[r][1] - spans[r][0], np.nan) for r in eval_runs}
    for r, t, sc in zip(win_run[eval_mask], win_t[eval_mask], dist.mean(axis=1)):
        topo[r][t] = sc
    reps = {r: rolling_mean(dup_flags(streamed[sub["id"][r]]["actions"]), ROLL) for r in eval_runs}

    s_topo, on_topo = alarm_onsets(topo, eval_runs, grp, cum, total, "topology (rolling P1)")
    s_rep, on_rep = alarm_onsets(reps, eval_runs, grp, cum, total, "repetition (rolling)")
    summary = pd.DataFrame([s_topo, s_rep])
    summary.to_csv(out_dir / "context_alarm_summary.csv", index=False)
    pd.concat([on_topo, on_rep]).to_csv(out_dir / "context_alarm_onsets.csv", index=False)

    print(f"\n=== 2/3) When do alarms fire in hit_limit runs? (threshold = {FALSE_ALARM:.0%} false alarms on successful runs) ===")
    print(summary.round(3).to_string(index=False))

    per_turn = []
    for r in eval_runs:
        n = spans[r][1] - spans[r][0]
        for t in range(n):
            per_turn.append({
                "run": int(r),
                "group": grp[r],
                "turns_before_end": n - 1 - t,
                "ctx_frac": cum[r][t] / total[r],
                "topo": topo[r][t],
                "rep": reps[r][t],
            })
    pt = pd.DataFrame(per_turn)
    pt["tbe_bin"] = pd.cut(pt["turns_before_end"], TBE_BINS, labels=TBE_LABELS)
    pt["ctx_bin"] = pd.cut(pt["ctx_frac"], CTX_BINS, labels=CTX_LABELS, right=False)
    pt.to_csv(out_dir / "context_per_turn.csv", index=False)

    print("\nMean rolling scores by turns before the end of the run:")
    print(pt.pivot_table(index="tbe_bin", columns="group", values=["topo", "rep"], aggfunc="mean", observed=False)
          .round(3).to_string())
    print("\nMean rolling scores by fraction of the run's total characters used so far:")
    print(pt.pivot_table(index="ctx_bin", columns="group", values=["topo", "rep"], aggfunc="mean", observed=False)
          .round(3).to_string())
    print(f"\nWrote context_p1_chars.csv, context_alarm_summary.csv, context_alarm_onsets.csv, context_per_turn.csv to {out_dir}")


if __name__ == "__main__":
    main()