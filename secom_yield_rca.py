"""
Semiconductor Yield Root-Cause Analysis on the UCI SECOM dataset.

Pipeline:
  1. Download SECOM (1,567 lots x 590 process signals, pass/fail label)
  2. Load into a relational database (SQLite) and mine it with SQL
  3. Clean: drop high-missing / constant signals, median-impute
  4. Root cause: Mann-Whitney U test per signal (pass vs fail), FDR-corrected
  5. SPC: X-bar +/- 3 sigma control charts for top signals
  6. Fail prediction: imbalance-aware Random Forest, stratified CV
  7. Write results/metrics.json + charts + summary to results/

Run:  pip install pandas numpy scipy scikit-learn matplotlib
      python secom_yield_rca.py
"""
import io
import json
import sqlite3
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"
DB = DATA / "secom.db"
URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"

MISSING_THRESHOLD = 0.50   # drop signals with >50% missing
ALPHA = 0.01               # significance level after FDR correction
TOP_K = 15                 # signals kept for the model / report
SEED = 42


# ---------------------------------------------------------------- 1. ingest
def download():
    DATA.mkdir(exist_ok=True)
    if (DATA / "secom.data").exists() and (DATA / "secom_labels.data").exists():
        return
    print("Downloading SECOM from UCI ...")
    with urllib.request.urlopen(URL, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    for name in ("secom.data", "secom_labels.data"):
        member = next(m for m in z.namelist() if m.endswith(name))
        (DATA / name).write_bytes(z.read(member))


def load_to_db():
    X = pd.read_csv(DATA / "secom.data", sep=r"\s+", header=None)
    X.columns = [f"s{i:03d}" for i in range(X.shape[1])]
    y = pd.read_csv(DATA / "secom_labels.data", sep=r"\s+", header=None,
                    names=["label", "ts"], quotechar='"')
    lots = pd.DataFrame({
        "lot_id": np.arange(len(y)),
        "ts": pd.to_datetime(y["ts"], dayfirst=True, errors="coerce"),
        "fail": (y["label"] == 1).astype(int),   # SECOM: -1 = pass, 1 = fail
    })
    # long/tidy table: one row per (lot, signal) -- how fab data usually lives
    meas = X.copy().assign(lot_id=lots["lot_id"]).melt(
        id_vars="lot_id", var_name="signal", value_name="value")
    with sqlite3.connect(DB) as con:
        lots.to_sql("lots", con, if_exists="replace", index=False)
        meas.to_sql("measurements", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_meas_sig ON measurements(signal)")
        con.execute("CREATE INDEX IF NOT EXISTS ix_meas_lot ON measurements(lot_id)")
    return len(lots), X.shape[1]


# ------------------------------------------------------- 2. SQL data mining
SQL_YIELD_BY_MONTH = """
SELECT strftime('%Y-%m', ts) AS month,
       COUNT(*)              AS lots,
       SUM(fail)             AS fails,
       ROUND(100.0 * (1 - AVG(fail)), 2) AS yield_pct
FROM lots GROUP BY month ORDER BY month;
"""

SQL_SIGNAL_PROFILE = """
SELECT m.signal,
       AVG(m.value IS NULL)                              AS missing_rate,
       COUNT(DISTINCT m.value)                           AS n_distinct,
       AVG(CASE WHEN l.fail = 0 THEN m.value END)        AS mean_pass,
       AVG(CASE WHEN l.fail = 1 THEN m.value END)        AS mean_fail
FROM measurements m JOIN lots l USING (lot_id)
GROUP BY m.signal;
"""


def mine(con):
    yield_month = pd.read_sql(SQL_YIELD_BY_MONTH, con)
    profile = pd.read_sql(SQL_SIGNAL_PROFILE, con)
    return yield_month, profile


# ---------------------------------------------------------------- 3. clean
def build_matrix(con, profile):
    keep = profile[(profile.missing_rate <= MISSING_THRESHOLD)
                   & (profile.n_distinct > 1)].signal.tolist()
    wide = (pd.read_sql("SELECT lot_id, signal, value FROM measurements", con)
              .pivot(index="lot_id", columns="signal", values="value"))[keep]
    wide = wide.loc[:, wide.std(skipna=True) > 0]
    wide = wide.fillna(wide.median())
    lots = pd.read_sql("SELECT * FROM lots ORDER BY lot_id", con, parse_dates=["ts"])
    return wide, lots


# ----------------------------------------------------------- 4. root cause
def bh_fdr(p):
    p = np.asarray(p)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def root_cause(X, fail):
    rows = []
    for col in X.columns:
        a, b = X.loc[fail == 0, col], X.loc[fail == 1, col]
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        # effect size: shift in median, in units of pass-lot std
        shift = (b.median() - a.median()) / (a.std() or 1)
        rows.append((col, p, shift))
    res = pd.DataFrame(rows, columns=["signal", "p_value", "shift_sigma"])
    res["q_value"] = bh_fdr(res.p_value)
    return res.sort_values("p_value").reset_index(drop=True)


# ------------------------------------------------------------------ 5. SPC
def control_charts(X, lots, signals):
    order = lots.sort_values("ts").lot_id.values
    fig, axes = plt.subplots(len(signals), 1, figsize=(11, 3.2 * len(signals)),
                             sharex=True)
    summary = {}
    for ax, sig in zip(np.atleast_1d(axes), signals):
        v = X.loc[order, sig].values
        f = lots.set_index("lot_id").loc[order, "fail"].values
        pass_v = X.loc[lots.fail == 0, sig]
        mu, sd = pass_v.mean(), pass_v.std()
        ucl, lcl = mu + 3 * sd, mu - 3 * sd
        ooc = (v > ucl) | (v < lcl)
        ax.plot(v, lw=0.6, color="#4a6fa5")
        ax.scatter(np.where(f == 1)[0], v[f == 1], s=10, color="#d1495b",
                   label="failed lot", zorder=3)
        for y_, ls in ((mu, "-"), (ucl, "--"), (lcl, "--")):
            ax.axhline(y_, color="#333", lw=0.8, ls=ls)
        ax.set_title(f"{sig}  |  out-of-control lots: {ooc.sum()}  "
                     f"| fail rate when OOC: {f[ooc].mean() if ooc.any() else 0:.0%}",
                     fontsize=10, loc="left")
        summary[sig] = {"ooc_lots": int(ooc.sum()),
                        "fail_rate_ooc": float(f[ooc].mean()) if ooc.any() else None,
                        "fail_rate_in_control": float(f[~ooc].mean())}
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("lot sequence (time-ordered)")
    fig.tight_layout()
    fig.savefig(OUT / "control_charts.png", dpi=150)
    plt.close(fig)
    return summary


# ------------------------------------------------------------- 6. predict
def make_rf():
    return RandomForestClassifier(n_estimators=400, max_depth=8,
                                  min_samples_leaf=3, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=SEED)


def predict(X, fail, top_k=None):
    """Stratified 5-fold CV. If top_k is set, signals are re-selected
    inside each training fold (Mann-Whitney) so there is no leakage."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rf = make_rf()
    proba = np.zeros(len(fail))
    for tr, te in cv.split(X, fail):
        cols = X.columns
        if top_k:
            cols = root_cause(X.iloc[tr], fail[tr]).head(top_k).signal
        m = make_rf().fit(X.iloc[tr][cols], fail[tr])
        proba[te] = m.predict_proba(X.iloc[te][cols])[:, 1]
    auc = roc_auc_score(fail, proba)
    ap = average_precision_score(fail, proba)
    # pick the threshold that catches >=70% of fails with the fewest flags
    best = None
    for t in np.linspace(0.05, 0.95, 91):
        pred = (proba >= t).astype(int)
        rec = recall_score(fail, pred)
        if rec >= 0.70:
            best = (t, rec, pred.mean())
    t, rec, flag_rate = best if best else (0.5, recall_score(fail, proba >= 0.5),
                                           (proba >= 0.5).mean())
    tn, fp, fn, tp = confusion_matrix(fail, proba >= t).ravel()
    final_cols = root_cause(X, fail).head(top_k).signal if top_k else X.columns
    rf.fit(X[final_cols], fail)
    imp = pd.Series(rf.feature_importances_, index=final_cols).sort_values(ascending=False)
    return {"roc_auc": round(auc, 3), "pr_auc": round(ap, 3),
            "baseline_fail_rate": round(float(fail.mean()), 3),
            "threshold": round(float(t), 2), "recall_fail": round(float(rec), 3),
            "pct_lots_flagged": round(float(flag_rate), 3),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}, imp


def plot_top_signals(rc, k):
    top = rc.head(k)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * k + 1))
    ax.barh(top.signal, -np.log10(top.p_value), color="#4a6fa5")
    ax.axvline(-np.log10(ALPHA), color="#d1495b", ls="--", lw=1, label=f"p = {ALPHA}")
    ax.set_xlabel("-log10(p)  (Mann-Whitney U, pass vs fail)")
    ax.set_title(f"Top {k} process signals separating failed lots", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "top_signals.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------- main
def main():
    OUT.mkdir(exist_ok=True)
    download()
    n_lots, n_raw = load_to_db()
    with sqlite3.connect(DB) as con:
        yield_month, profile = mine(con)
        X, lots = build_matrix(con, profile)
    fail = lots.fail.values

    rc = root_cause(X, fail)
    sig = rc[rc.q_value < ALPHA]
    top = rc.head(TOP_K).signal.tolist()
    plot_top_signals(rc, TOP_K)
    spc = control_charts(X, lots, top[:3])

    model_all, imp = predict(X, fail)
    model_top, _ = predict(X, fail, top_k=TOP_K)

    metrics = {
        "lots": n_lots, "raw_signals": n_raw, "signals_after_cleaning": X.shape[1],
        "signals_removed": n_raw - X.shape[1],
        "fail_lots": int(fail.sum()), "yield_pct": round(100 * (1 - fail.mean()), 2),
        "significant_signals_q<0.01": int(len(sig)),
        "top_signals": top,
        "spc_top3": spc,
        "model_all_signals": model_all,
        f"model_top{TOP_K}_signals": model_top,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    rc.to_csv(OUT / "signal_ranking.csv", index=False)
    yield_month.to_csv(OUT / "yield_by_month.csv", index=False)
    imp.head(25).to_csv(OUT / "rf_feature_importance.csv", header=["importance"])

    print("\n=== SECOM Yield RCA ===")
    print(f"Lots: {n_lots} | raw signals: {n_raw} -> {X.shape[1]} after cleaning")
    print(f"Fails: {fail.sum()} ({fail.mean():.1%}) | yield {metrics['yield_pct']}%")
    print(f"Significant signals (FDR q<{ALPHA}): {len(sig)}")
    print(f"Top 3: {top[:3]}")
    for k, m in (("all signals", model_all), (f"top {TOP_K}", model_top)):
        print(f"RF [{k}]  ROC-AUC {m['roc_auc']} | recall on fails {m['recall_fail']:.0%} "
              f"flagging {m['pct_lots_flagged']:.0%} of lots")
    print(f"\nOutputs in {OUT}")


if __name__ == "__main__":
    main()
