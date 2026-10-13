"""Do PH features of reasoning traces predict failure beyond length-type baselines?"""
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from persim import plot_diagrams
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from trace_features import cosine_distance_matrix, persistence_diagrams


def make_classifier(name, seed):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=seed)
    raise ValueError(f"unknown classifier {name}")


def build_feature_sets(cfg):
    groups = cfg["feature_groups"]
    sets = {}
    for name, members in cfg["feature_sets"].items():
        cols = []
        for group in members:
            for col in groups[group]:
                if col not in cols:
                    cols.append(col)
        sets[name] = cols
    return sets


def make_targets(df):
    return {
        "final_answer_wrong": (~df["final_answer_correct"]).astype(int).to_numpy(),
        "process_error": (df["label"] != -1).astype(int).to_numpy(),
    }


def oof_scores(clf, X, y, folds, seed, groups=None):
    if groups is None:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        return cross_val_predict(clone(clf), X, y, cv=cv, method="predict_proba")[:, 1]
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(clone(clf), X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]


def bootstrap_aucs(y, score_sets, n_boot, seed, desc):
    rng = np.random.default_rng(seed)
    n = len(y)
    out = {name: [] for name in score_sets}
    for _ in tqdm(range(n_boot), desc=desc, unit="resample", leave=False):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for name, s in score_sets.items():
            out[name].append(roc_auc_score(yb, s[idx]))
    return {name: np.asarray(v) for name, v in out.items()}


def summarize(point, boots):
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auc": float(point), "ci_low": float(lo), "ci_high": float(hi)}


def delta_summaries(y, scores, boots):
    base = roc_auc_score(y, scores["baseline"])
    out = {}
    for name, s in scores.items():
        if name.endswith("+baseline"):
            lo, hi = np.percentile(boots[name] - boots["baseline"], [2.5, 97.5])
            out[name] = {"delta_auc": float(roc_auc_score(y, s) - base), "ci_low": float(lo), "ci_high": float(hi)}
    return out


def block_from_scores(y, scores, cfg, seed, desc):
    boots = bootstrap_aucs(y, scores, cfg["n_bootstrap"], seed, desc)
    return {
        "cv": {name: summarize(roc_auc_score(y, s), boots[name]) for name, s in scores.items()},
        "deltas": delta_summaries(y, scores, boots),
    }


def evaluate_block(df, y, feature_sets, clf_name, cfg, seed, desc, groups=None):
    scores = {
        name: oof_scores(make_classifier(clf_name, seed), df[cols].to_numpy(dtype=float), y, cfg["cv_folds"], seed, groups)
        for name, cols in feature_sets.items()
    }
    return scores, block_from_scores(y, scores, cfg, seed, desc)


def plot_auc_bars(cv, path, title):
    names = list(cv)
    aucs = [cv[n]["auc"] for n in names]
    err = np.array([[cv[n]["auc"] - cv[n]["ci_low"] for n in names], [cv[n]["ci_high"] - cv[n]["auc"] for n in names]])
    colors = plt.cm.tab10(np.arange(len(names)) % 10)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(names, aucs, yerr=err, capsize=5, color=colors)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("ROC AUC (out-of-fold)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pick_examples(df, y, min_steps):
    eligible = (df["n_steps"] >= min_steps).to_numpy()
    wrong = df[eligible & (y == 1)]
    right = df[eligible & (y == 0)]
    w = wrong.sort_values("h1_total", ascending=False).iloc[0]
    same_len = right[right["n_steps"] == w["n_steps"]]
    pool = same_len if len(same_len) else right
    c = pool.sort_values("h1_total", ascending=True).iloc[0]
    return w["id"], c["id"]


def plot_trace(ax_path, ax_dgm, X, title):
    P = PCA(n_components=2, random_state=0).fit_transform(X)
    ax_path.plot(P[:, 0], P[:, 1], "-", color="#888888", linewidth=1, zorder=1)
    ax_path.scatter(P[:, 0], P[:, 1], c=np.arange(len(P)), cmap="viridis", s=30, zorder=2)
    for i, (px, py) in enumerate(P):
        ax_path.annotate(str(i + 1), (px, py), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax_path.set_title(title)
    ax_path.set_xticks([])
    ax_path.set_yticks([])
    dgms = persistence_diagrams(cosine_distance_matrix(X), maxdim=1)
    plot_diagrams(dgms, ax=ax_dgm, show=False)
    ax_dgm.set_title("persistence diagram (cosine)")


def print_block(label, block):
    aucs = "  ".join(f"{n} {r['auc']:.3f}" for n, r in block["cv"].items())
    print(f"  {label}")
    print(f"    AUC: {aucs}")
    for name, d in block["deltas"].items():
        print(f"    {name:26s} adds {d['delta_auc']:+.3f}  [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    seed = cfg["seed"]
    tag = cfg["tag"]
    data_dir = Path(cfg["data_dir"])
    out_dir = Path(cfg["results_dir"]) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_dir / f"features_{tag}.parquet")
    feature_sets = build_feature_sets(cfg)
    all_feature_cols = []
    for cols in cfg["feature_groups"].values():
        for c in cols:
            if c not in all_feature_cols:
                all_feature_cols.append(c)

    all_targets = make_targets(df)
    wanted_targets = cfg.get("targets") or list(all_targets)
    targets = {k: all_targets[k] for k in wanted_targets}
    primary = cfg["primary_target"]
    primary_clf = cfg["primary_classifier"]
    group_col = cfg.get("group_col")
    groups = df[group_col].to_numpy() if group_col else None
    splits = cfg.get("splits") or [s for s, c in df["split"].value_counts().items() if c >= cfg.get("min_split_n", 150)]
    length_cols = [c for c in ["n_steps", "n_points", "n_chars", "n_messages"] if c in df.columns]

    results = {
        "tag": tag,
        "config": cfg_path,
        "n_traces": int(len(df)),
        "group_col": group_col,
        "primary_target": primary,
        "primary_classifier": primary_clf,
        "splits": splits,
        "feature_sets": feature_sets,
        "median_points_by_split": {s: float(df.loc[df["split"] == s, "n_points"].median()) for s in splits},
        "h1_presence_by_split": {s: float((df.loc[df["split"] == s, "h1_count"] > 0).mean()) for s in splits},
        "h2_presence_by_split": {s: float((df.loc[df["split"] == s, "count_h2"] > 0).mean()) for s in splits},
        "prevalence": {},
        "pooled": {},
        "within_split": {},
        "univariate_auc": {},
        "length_check": {},
        "per_generator": {},
        "transfer": {},
    }

    primary_scores = None
    for tname, y in targets.items():
        results["prevalence"][tname] = float(y.mean())
        results["pooled"][tname] = {}
        for clf_name in cfg["classifiers"]:
            scores, block = evaluate_block(df, y, feature_sets, clf_name, cfg, seed, f"pooled {tname} {clf_name}", groups)
            results["pooled"][tname][clf_name] = block
            if tname == primary and clf_name == primary_clf:
                primary_scores = scores
        results["univariate_auc"][tname] = {c: float(roc_auc_score(y, df[c].to_numpy(dtype=float))) for c in all_feature_cols}
        results["length_check"][tname] = {}
        for c in length_cols:
            r = spearmanr(df[c].to_numpy(dtype=float), y)
            results["length_check"][tname][c] = {"spearman_rho": float(r.statistic), "p_value": float(r.pvalue)}

    y = targets[primary]
    for split in tqdm(splits, desc="within-split", unit="split"):
        mask = (df["split"] == split).to_numpy()
        ys = y[mask]
        if ys.min() == ys.max():
            continue
        dfs = df[mask].reset_index(drop=True)
        gs = dfs[group_col].to_numpy() if group_col else None
        results["within_split"][split] = {}
        for clf_name in cfg["classifiers"]:
            _, block = evaluate_block(dfs, ys, feature_sets, clf_name, cfg, seed, f"{split} {clf_name}", gs)
            results["within_split"][split][clf_name] = block

    for gen, idx in df.groupby("generator").indices.items():
        yg = y[idx]
        if len(idx) < cfg["min_generator_n"] or yg.min() == yg.max():
            continue
        results["per_generator"][gen] = {
            "n": int(len(idx)),
            "prevalence": float(yg.mean()),
            **{name: float(roc_auc_score(yg, s[idx])) for name, s in primary_scores.items()},
        }

    for pair in cfg.get("transfer") or []:
        tr = (df["split"] == pair["train"]).to_numpy()
        te = (df["split"] == pair["test"]).to_numpy()
        if tr.sum() == 0 or te.sum() == 0:
            continue
        ytr, yte = y[tr], y[te]
        if ytr.min() == ytr.max() or yte.min() == yte.max():
            continue
        key = f"{pair['train']}->{pair['test']}"
        results["transfer"][key] = {}
        for clf_name in cfg["classifiers"]:
            scores = {}
            for name, cols in feature_sets.items():
                model = make_classifier(clf_name, seed)
                model.fit(df.loc[tr, cols].to_numpy(dtype=float), ytr)
                scores[name] = model.predict_proba(df.loc[te, cols].to_numpy(dtype=float))[:, 1]
            results["transfer"][key][clf_name] = block_from_scores(yte, scores, cfg, seed, f"transfer {key} {clf_name}")

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    plot_auc_bars(
        results["pooled"][primary][primary_clf]["cv"],
        out_dir / "auc_cv.png",
        f"Out-of-fold AUC ({primary_clf}, {tag}), target = {primary}",
    )

    with open(data_dir / f"traces_{tag}.pkl", "rb") as f:
        traces = pickle.load(f)
    wrong_id, right_id = pick_examples(df, y, cfg["min_steps_for_examples"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    plot_trace(axes[0, 0], axes[0, 1], traces[wrong_id]["X"], f"Failed trace {wrong_id} (highest H1 total)")
    plot_trace(axes[1, 0], axes[1, 1], traces[right_id]["X"], f"Successful trace {right_id} (lowest H1 total, same steps)")
    fig.suptitle(f"Illustrative extremes, not typical cases ({tag})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "example_traces.png", dpi=150)
    plt.close(fig)

    print(f"\n=== {tag} | primary target: {primary} (prevalence {results['prevalence'][primary]:.2f}) | grouped by: {group_col} ===")
    print("\nPooled:")
    for clf_name, block in results["pooled"][primary].items():
        print_block(clf_name, block)
    print("\nWithin split:")
    for split, blocks in results["within_split"].items():
        for clf_name, block in blocks.items():
            print_block(f"{split} / {clf_name}", block)
    print("\nLength check (Spearman with target):")
    for c, r in results["length_check"][primary].items():
        print(f"  {c:10s} rho {r['spearman_rho']:+.3f}  p {r['p_value']:.3g}")
    print("\nUnivariate AUC (>0.5 means higher value -> more likely wrong):")
    for c, a in results["univariate_auc"][primary].items():
        print(f"  {c:22s} {a:.3f}")
    print("\nMedian points, H1 loop presence, H2 void presence:")
    for s in splits:
        print(
            f"  {s:14s} median points {results['median_points_by_split'][s]:6.1f}   "
            f"H1 {results['h1_presence_by_split'][s]:.2f}   H2 {results['h2_presence_by_split'][s]:.2f}"
        )

if __name__ == "__main__":
    main()