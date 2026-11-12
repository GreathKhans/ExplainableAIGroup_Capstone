"""Explanation stability check for the Credit Risk Auditor.

Fidelity (base value + SHAP = log-odds) shows the explanation is exact for ONE
fitted model. It does not show that the explanation would survive a slightly
different training sample, or a different SHAP background. This script tests
both, on the four demo applicants, using the same encoding, split and model
settings as auditor/pipeline.py.

Run from the repository root:  python analysis/explanation_stability.py
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auditor.pipeline import (EXAMPLE_ROWS, RANDOM_STATE, SCORING_PATH,
                              load_frame)
from auditor.codebook import RAW_COLUMNS

N_RUNS = 30            # bootstrap retrains
TOP_K = 3
STABLE_SHARE = 0.80    # illustrative project threshold, not a regulatory value

frame = load_frame()
encoded = pd.get_dummies(frame[RAW_COLUMNS], drop_first=True)
x_train, x_test, y_train, y_test = train_test_split(
    encoded, frame["target"], test_size=0.2, random_state=RANDOM_STATE)

col_to_field = {}
for col in encoded.columns:
    hits = [f for f in RAW_COLUMNS if col == f or col.startswith(f + "_")]
    col_to_field[col] = max(hits, key=len)

def grouped(values):
    return pd.Series(values, index=encoded.columns).groupby(col_to_field).sum()

def top_up(g, k=TOP_K):
    g = g[g > 0].sort_values(ascending=False)
    return list(g.index[:k])

saved = xgb.XGBClassifier()
saved.load_model(SCORING_PATH)
rows = encoded.loc[list(EXAMPLE_ROWS)].astype(float)

# ---- Fidelity on the saved model ----
expl = shap.TreeExplainer(saved)
sv = expl(rows)
margin = saved.predict(rows, output_margin=True)
fidelity_err = float(np.max(np.abs(sv.base_values + sv.values.sum(1) - margin)))
base = {r: grouped(sv.values[i]) for i, r in enumerate(EXAMPLE_ROWS)}
base_prob = dict(zip(EXAMPLE_ROWS, saved.predict_proba(rows)[:, 1]))

# ---- Bootstrap retraining ----
rng = np.random.default_rng(RANDOM_STATE)
runs = {r: [] for r in EXAMPLE_ROWS}
probs = {r: [] for r in EXAMPLE_ROWS}
global_rank_corr = []
base_global = grouped(np.abs(expl(x_test.astype(float)).values).mean(0))
t0 = time.perf_counter()
for i in range(N_RUNS):
    idx = rng.choice(len(x_train), len(x_train), replace=True)
    m = xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss",
                          max_depth=4, device="cpu")
    m.fit(x_train.iloc[idx], y_train.iloc[idx])
    e = shap.TreeExplainer(m)
    v = e(rows).values
    p = m.predict_proba(rows)[:, 1]
    for j, r in enumerate(EXAMPLE_ROWS):
        runs[r].append(grouped(v[j]))
        probs[r].append(p[j])
    g_glob = grouped(np.abs(e(x_test.astype(float)).values).mean(0))
    global_rank_corr.append(spearmanr(base_global, g_glob.reindex(base_global.index)).correlation)
boot_seconds = time.perf_counter() - t0

# ---- Background sensitivity (interventional, two backgrounds) ----
bg_tops = {}
for name, seed in [("bg_A", 1), ("bg_B", 2)]:
    bg = shap.sample(x_train.astype(float), 100, random_state=seed)
    ie = shap.TreeExplainer(saved, data=bg, feature_perturbation="interventional")
    iv = ie(rows).values
    bg_tops[name] = {r: top_up(grouped(iv[j])) for j, r in enumerate(EXAMPLE_ROWS)}

# ---- Report ----
print(f"Fidelity: max |base + sum(SHAP) - log-odds| = {fidelity_err:.2e}")
print(f"Bootstrap: {N_RUNS} retrains in {boot_seconds:.1f}s")
print(f"Global importance ranking, median Spearman vs saved model: {np.median(global_rank_corr):.3f} "
      f"(min {np.min(global_rank_corr):.3f})\n")
summary = []
for r in EXAMPLE_ROWS:
    top = top_up(base[r])
    share = {f: np.mean([f in top_up(g) for g in runs[r]]) for f in top}
    first_same = np.mean([top_up(g)[:1] == top[:1] for g in runs[r]])
    rc = np.median([spearmanr(base[r], g.reindex(base[r].index)).correlation for g in runs[r]])
    flag = np.mean(np.array(probs[r]) >= 0.5)
    print(f"Row {r}: saved model {base_prob[r]:.1%}, bootstrap {np.min(probs[r]):.1%}-{np.max(probs[r]):.1%}, "
          f"flagged higher risk in {flag:.0%} of runs")
    print(f"  saved top-{TOP_K}: {top}   path-dependent")
    print(f"  bg_A top-{TOP_K}:  {bg_tops['bg_A'][r]}   interventional")
    print(f"  bg_B top-{TOP_K}:  {bg_tops['bg_B'][r]}   interventional")
    print(f"  share of retrains keeping each factor in top-{TOP_K}: " +
          ", ".join(f"{f} {s:.0%}" for f, s in share.items()))
    print(f"  same #1 factor in {first_same:.0%} of retrains; median rank corr {rc:.3f}")
    for f, s in share.items():
        summary.append({"row": r, "factor": f, "share_top3": round(s, 2),
                        "stable": s >= STABLE_SHARE,
                        "in_both_backgrounds": f in bg_tops["bg_A"][r] and f in bg_tops["bg_B"][r]})
print()
print(pd.DataFrame(summary).to_string(index=False))
