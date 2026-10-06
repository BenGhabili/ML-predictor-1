"""
modules/orderflow_sweep.py
Sweep all (target, horizon) combos and rank by estimated profitability.

Usage:
    python -m modules.orderflow_sweep --data-dir data_orderflow --symbol NQ
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit

from services.model_factory import build_model
from services.train_utils import load_processed_csv

# ── symbol cost table ────────────────────────────────────────────────
SYMBOL_COSTS = {
    "NQ": {
        "tick_size": 0.25,
        "tick_value_usd": 5.0,
        "fees_roundtrip_usd": 4.5,
        "slippage_ticks_per_side": 0.5,
        "cost_ticks": 1.9,
    },
    "ES": {
        "tick_size": 0.25,
        "tick_value_usd": 12.50,
        "fees_roundtrip_usd": 5.0,
        "slippage_ticks_per_side": 0.5,
        "cost_ticks": 1.4,
    },
}

# ── filename → (target, horizon) parser ──────────────────────────────
_RE_TH = re.compile(r"_t([\d.]+)_h(\d+)")


def parse_target_horizon(fname: str) -> tuple[float, int] | None:
    """Extract (target_net_ticks, horizon_s) from a compact CSV filename."""
    m = _RE_TH.search(fname)
    if not m:
        return None
    return float(m.group(1)), int(m.group(2))


# ── per-combo train + evaluate ───────────────────────────────────────
def evaluate_combo(
    csv_path: Path,
    target: float,
    horizon: int,
    symbol: str,
    algo: str,
    cv_mode: str,
    trees: int,
    depth: int,
    eta: float,
    save_models: bool,
    max_rows: int | None = None,
) -> dict:
    """Train walk-forward CV on one CSV, return summary dict."""
    X, y = load_processed_csv(csv_path)

    # subsample: keep the LAST max_rows (most recent, preserves time order)
    if max_rows and len(y) > max_rows:
        print(f"  Subsampling {len(y):,} → {max_rows:,} rows (most recent)")
        X = X[-max_rows:]
        y = y[-max_rows:]

    n_rows = len(y)

    # class distribution
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique.astype(int), counts))
    total = counts.sum()
    pct_flat = dist.get(0, 0) / total
    pct_up = dist.get(1, 0) / total
    pct_dn = dist.get(2, 0) / total

    # CV splitter
    if cv_mode == "walk":
        n_splits = 5
        test_size = n_rows // (n_splits + 1)
        if test_size < 50:
            n_splits = max(2, n_rows // 100)
            test_size = n_rows // (n_splits + 1)
        cv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        cv_iter = list(cv.split(X))
    else:
        from sklearn.model_selection import train_test_split
        tr_idx, te_idx = train_test_split(
            np.arange(n_rows), test_size=0.25, shuffle=True, random_state=42)
        cv_iter = [(tr_idx, te_idx)]

    f1_scores = []
    last_report = {}
    last_y_pred = None
    last_y_test = None

    for tr_idx, te_idx in cv_iter:
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # skip folds with fewer than 2 classes in train set
        if len(np.unique(y_tr)) < 2:
            continue

        try:
            model = build_model(
                algo, X_tr, y_tr, False,
                trees=trees, depth=depth, eta=eta,
                X_val=X_te, y_val=y_te,
            )
            y_pred = model.predict(X_te)
            # ensure predictions are flat class labels (not probabilities)
            if y_pred.ndim > 1:
                y_pred = np.argmax(y_pred, axis=1)
            f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
            f1_scores.append(f1)
            last_y_pred = y_pred
            last_y_test = y_te
        except Exception as e:
            print(f"  WARNING: fold failed ({e}), skipping")
            continue

    # classification report from last fold
    last_report = classification_report(
        last_y_test, last_y_pred, output_dict=True, zero_division=0)

    # extract per-class metrics (classes: 0=flat, 1=up, 2=dn)
    prec_up = last_report.get("1", last_report.get("1.0", {})).get("precision", 0)
    prec_dn = last_report.get("2", last_report.get("2.0", {})).get("precision", 0)
    recall_up = last_report.get("1", last_report.get("1.0", {})).get("recall", 0)
    recall_dn = last_report.get("2", last_report.get("2.0", {})).get("recall", 0)

    # profitability estimate
    costs = SYMBOL_COSTS.get(symbol.upper(), SYMBOL_COSTS["NQ"])
    tick_value_usd = costs["tick_value_usd"]

    # signals per day: predicted non-flat in last fold, scaled to full dataset
    n_signals_last = np.sum(last_y_pred != 0)
    n_test_last = len(last_y_test)

    # estimate trading days from row count and typical session length
    # rough: ~6.5h RTH ≈ 23400s, with tick data arriving ~10-50/s
    # better: use horizon to estimate decision points per day
    # simplest: count signals in test fold, scale by (total rows / test rows)
    signal_rate = n_signals_last / n_test_last if n_test_last > 0 else 0
    total_signals = signal_rate * n_rows

    # estimate trading days (assume ~250 trading days/year, scale by data volume)
    # use the CSV's row count and a rough ticks-per-day estimate
    # conservative: 20k-50k ticks per RTH day for NQ
    ticks_per_day_est = 30_000
    trading_days = max(1, n_rows / ticks_per_day_est)
    signals_per_day = total_signals / trading_days

    # EV calculation
    # win_rate = weighted precision across directional classes
    n_up_pred = np.sum(last_y_pred == 1)
    n_dn_pred = np.sum(last_y_pred == 2)
    n_directional = n_up_pred + n_dn_pred
    if n_directional > 0:
        win_rate = (prec_up * n_up_pred + prec_dn * n_dn_pred) / n_directional
    else:
        win_rate = 0.0

    target_net = target  # already net of costs from labelling
    ev_per_trade_ticks = win_rate * target_net - (1 - win_rate) * target_net
    daily_ev_ticks = ev_per_trade_ticks * signals_per_day
    daily_ev_usd = daily_ev_ticks * tick_value_usd

    # optionally save model trained on full data
    if save_models:
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        full_model = build_model(
            algo, X, y, False,
            trees=trees, depth=depth, eta=eta,
        )
        from joblib import dump
        out_path = model_dir / f"{algo}_of_t{target}_h{horizon}.pkl"
        dump(full_model, out_path)
        print(f"  Saved model → {out_path}")

    return {
        "symbol": symbol.upper(),
        "target": target,
        "horizon": horizon,
        "rows": n_rows,
        "f1_mean": np.mean(f1_scores),
        "f1_std": np.std(f1_scores),
        "pct_flat": round(pct_flat, 4),
        "pct_up": round(pct_up, 4),
        "pct_dn": round(pct_dn, 4),
        "prec_up": round(prec_up, 4),
        "prec_dn": round(prec_dn, 4),
        "recall_up": round(recall_up, 4),
        "recall_dn": round(recall_dn, 4),
        "win_rate": round(win_rate, 4),
        "signals_per_day": round(signals_per_day, 1),
        "ev_per_trade_ticks": round(ev_per_trade_ticks, 4),
        "daily_ev_usd": round(daily_ev_usd, 2),
    }


# ── discover CSVs via manifest or glob fallback ─────────────────────
def _discover_combos(data_dir: Path) -> list[tuple[Path, float, int]]:
    """Return [(csv_path, target, horizon), ...] from manifest or glob.

    Priority: manifest.csv (written by of_prep) → glob fallback.
    """
    manifest_path = data_dir / "manifest.csv"
    if manifest_path.exists():
        mf = pd.read_csv(manifest_path)
        combos = []
        for _, row in mf.iterrows():
            csv_path = data_dir / row["file"]
            if not csv_path.exists():
                print(f"WARNING: manifest lists {row['file']} but file not found, skipping")
                continue
            combos.append((csv_path, float(row["target_net_ticks"]), int(row["horizon_s"])))
        if combos:
            print(f"Using manifest.csv ({len(combos)} combos)")
            return combos
        print("WARNING: manifest.csv is empty or all files missing, falling back to glob")

    # fallback: glob + parse filenames (check both naming patterns)
    csvs = sorted(data_dir.glob("orderflow_compact_*.csv"))
    csvs += sorted(data_dir.glob("l2_compact_*.csv"))
    combos = []
    for csv_path in csvs:
        parsed = parse_target_horizon(csv_path.name)
        if parsed is None:
            print(f"Skipping {csv_path.name} (cannot parse target/horizon)")
            continue
        combos.append((csv_path, parsed[0], parsed[1]))
    if combos:
        print(f"No manifest found — discovered {len(combos)} combos via glob")
    return combos


# ── main sweep ───────────────────────────────────────────────────────
def run_sweep(
    data_dir: Path,
    symbol: str,
    algo: str,
    cv: str,
    trees: int,
    depth: int,
    eta: float,
    save_models: bool,
    max_rows: int | None = None,
) -> pd.DataFrame:
    combos = _discover_combos(data_dir)
    if not combos:
        raise FileNotFoundError(f"No orderflow_compact_*.csv files in {data_dir}")

    results = []
    for csv_path, target, horizon in combos:
        print(f"\n{'='*60}")
        print(f"  {symbol.upper()} | target={target} ticks | horizon={horizon}s")
        print(f"  File: {csv_path.name}")
        print(f"{'='*60}")

        try:
            row = evaluate_combo(
                csv_path, target, horizon, symbol,
                algo, cv, trees, depth, eta, save_models,
                max_rows=max_rows,
            )
        except Exception as e:
            print(f"  ERROR: {e} — skipping this combo")
            continue
        results.append(row)
        print(f"  → F1={row['f1_mean']:.3f}±{row['f1_std']:.3f}  "
              f"signals/day={row['signals_per_day']:.0f}  "
              f"daily_ev=${row['daily_ev_usd']:.2f}")

    df = pd.DataFrame(results)
    if df.empty:
        print("No results to summarise.")
        return df

    df = df.sort_values("daily_ev_usd", ascending=False).reset_index(drop=True)

    # save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"sweep_{symbol.upper()}_{ts}.csv"
    df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"  SWEEP SUMMARY — {symbol.upper()} ({algo}, cv={cv})")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\nSaved → {out_path}")
    return df


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Sweep orderflow (target, horizon) combos and rank by profitability")
    p.add_argument("--data-dir", default="data_orderflow",
                   help="Directory with orderflow_compact_*.csv files")
    p.add_argument("--algo", default="xgb", choices=["xgb", "rf", "knn"])
    p.add_argument("--cv", default="walk", choices=["walk", "random"])
    p.add_argument("--trees", type=int, default=800)
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--eta", type=float, default=0.03)
    p.add_argument("--symbol", default="NQ", help="Symbol for cost lookup (NQ or ES)")
    p.add_argument("--save-models", action="store_true",
                   help="Save full-data model for each combo")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Max rows per combo (takes most recent N rows)")
    args = p.parse_args()

    run_sweep(
        data_dir=Path(args.data_dir),
        symbol=args.symbol,
        algo=args.algo,
        cv=args.cv,
        trees=args.trees,
        depth=args.depth,
        eta=args.eta,
        save_models=args.save_models,
        max_rows=args.max_rows,
    )
