"""
modules/l2_data_prep.py
Parse L2 nrdtocsv files, compute features, apply labels, emit compact CSVs.

Supports:
  --from-cache   Skip raw parsing; load a previously saved parquet cache
  --bar-size N   Aggregate per-trade data into N-second bars before labelling
  --grid         Force full cross-product of targets × horizons

Usage:
    # First run: parse raw data, save cache
    python -m modules.l2_data_prep \
        --input-dir /path/to/nrd-to-csv \
        --out-dir data_l2 \
        --targets 1,2,4,8 --horizons 5,10,20,30 \
        --symbol NQ --session-filter 08:30-15:00

    # Re-run with bar aggregation (fast, uses cache)
    python -m modules.l2_data_prep \
        --from-cache data_l2/l2_trades_cache.parquet \
        --out-dir data_l2_5s --bar-size 5 \
        --targets 2,4,8 --horizons 10,20,30 --grid \
        --symbol NQ
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from services.l2_parser import (
    parse_l2_file, select_l2_features,
    aggregate_to_bars, select_bar_features,
)
from services.orderflow_labels import make_labels_multi


# Cost defaults per symbol
SYMBOL_COSTS = {
    "NQ": {
        "tick_size": 0.25,
        "tick_value_usd": 5.0,
        "fees_roundtrip_usd": 4.5,
        "slippage_ticks_per_side": 0.5,
        "spread_cap_ticks": 8,
        "entry_delay_ms": 200,
    },
    "ES": {
        "tick_size": 0.25,
        "tick_value_usd": 12.50,
        "fees_roundtrip_usd": 5.0,
        "slippage_ticks_per_side": 0.5,
        "spread_cap_ticks": 4,
        "entry_delay_ms": 200,
    },
}


def _parse_list(s: str, cast=float) -> list:
    return [cast(x.strip()) for x in s.split(',') if x.strip()]


def main():
    p = argparse.ArgumentParser(description="L2 data prep: parse depth data → compact feature CSVs")
    p.add_argument("--input-dir", default=None, help="Directory with nrdtocsv *.csv files")
    p.add_argument("--from-cache", default=None, help="Path to parquet cache (skip raw parsing)")
    p.add_argument("--out-dir", default="data_l2", help="Output directory")
    p.add_argument("--targets", default="1,2,4,8", help="Net tick targets (comma list)")
    p.add_argument("--horizons", default="5,10,20,30", help="Horizon seconds (comma list)")
    p.add_argument("--symbol", default="NQ", help="Symbol for cost parameters")
    p.add_argument("--session-filter", default=None,
                   help="Filter to RTH hours (e.g. '08:30-15:00' in exchange tz)")
    p.add_argument("--bar-size", type=int, default=0,
                   help="Aggregate into N-second bars (0 = per-trade, default)")
    p.add_argument("--grid", action="store_true",
                   help="Force full cross-product of targets × horizons")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = _parse_list(args.targets, float)
    horizons = _parse_list(args.horizons, int)
    costs = SYMBOL_COSTS.get(args.symbol.upper(), SYMBOL_COSTS["NQ"])

    # ── Load data ────────────────────────────────────────────
    if args.from_cache:
        cache_path = Path(args.from_cache)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found: {cache_path}")
        print(f"Loading cache: {cache_path}")
        if cache_path.suffix == ".parquet":
            df_all = pd.read_parquet(cache_path)
        else:
            df_all = pd.read_csv(cache_path, index_col="Datetime", parse_dates=True)
        print(f"Loaded {len(df_all):,} trades from cache")
    else:
        if not args.input_dir:
            raise ValueError("Must specify --input-dir or --from-cache")
        input_dir = Path(args.input_dir)
        csv_files = sorted(input_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files in {input_dir}")

        print(f"Found {len(csv_files)} day files")
        print(f"Symbol: {args.symbol.upper()}")
        print(f"Targets: {targets}")
        print(f"Horizons: {horizons}")
        print(f"Costs: {costs}")

        day_dfs = []
        for csv_path in csv_files:
            print(f"\nParsing {csv_path.name}...")
            df = parse_l2_file(csv_path)
            if df.empty:
                print(f"  No trades found, skipping")
                continue
            print(f"  {len(df):,} trades with features")
            day_dfs.append(df)

        if not day_dfs:
            raise ValueError("No trade data found in any file")

        df_all = pd.concat(day_dfs, axis=0).sort_index()
        print(f"\nTotal: {len(df_all):,} trades across {len(day_dfs)} days")

        # session filter
        if args.session_filter:
            start_h, end_h = args.session_filter.split('-')
            sh, sm = int(start_h.split(':')[0]), int(start_h.split(':')[1])
            eh, em = int(end_h.split(':')[0]), int(end_h.split(':')[1])
            start_min = sh * 60 + sm
            end_min = eh * 60 + em
            idx_min = df_all.index.hour * 60 + df_all.index.minute
            mask = (idx_min >= start_min) & (idx_min < end_min)
            df_all = df_all[mask]
            print(f"After session filter ({args.session_filter}): {len(df_all):,} trades")

        # save cache for future runs (avoids re-parsing raw L2 data)
        try:
            cache_path = out_dir / "l2_trades_cache.parquet"
            df_all.to_parquet(cache_path)
        except ImportError:
            cache_path = out_dir / "l2_trades_cache.csv"
            df_all.to_csv(cache_path)
        print(f"Cache saved: {cache_path} ({cache_path.stat().st_size / 1e6:.1f} MB)")

    # ── Aggregate to bars if requested ───────────────────────
    if args.bar_size > 0:
        print(f"\nAggregating to {args.bar_size}s bars...")
        df_bars = aggregate_to_bars(df_all, args.bar_size)
        print(f"  {len(df_all):,} trades → {len(df_bars):,} bars")
        features = select_bar_features(df_bars)
        df_for_labels = df_bars[["last", "spread"]].copy()
    else:
        features = select_l2_features(df_all)
        df_for_labels = df_all[["last", "spread"]].copy()

    print(f"\nFeatures ({len(features.columns)}): {list(features.columns)}")

    df_for_labels.columns = ["Last", "Spread"]

    # ── Build target × horizon grid ──────────────────────────
    if args.grid:
        # force cross product
        pairs = [(t, H) for t in targets for H in horizons]
        # make_labels_multi needs lists that produce the right pairing;
        # easiest: call it with single-element lists per pair
    else:
        pairs = None  # let make_labels_multi handle pairing

    if pairs:
        # manual grid: call make_labels_multi per pair
        labels = {}
        for t, H in pairs:
            result = make_labels_multi(
                df_for_labels,
                targets_net_ticks=[t],
                horizons_s=[H],
                tick_size=costs["tick_size"],
                tick_value_usd=costs["tick_value_usd"],
                fees_roundtrip_usd=costs["fees_roundtrip_usd"],
                slippage_ticks_per_side=costs["slippage_ticks_per_side"],
                entry_delay_ms=costs["entry_delay_ms"],
                spread_cap_ticks=costs["spread_cap_ticks"],
            )
            labels.update(result)
        print(f"\nGenerating labels for {len(pairs)} target × horizon pairs (grid)...")
    else:
        print(f"\nGenerating labels for {len(targets)} targets × {len(horizons)} horizons...")
        labels = make_labels_multi(
            df_for_labels,
            targets_net_ticks=targets,
            horizons_s=horizons,
            tick_size=costs["tick_size"],
            tick_value_usd=costs["tick_value_usd"],
            fees_roundtrip_usd=costs["fees_roundtrip_usd"],
            slippage_ticks_per_side=costs["slippage_ticks_per_side"],
            entry_delay_ms=costs["entry_delay_ms"],
            spread_cap_ticks=costs["spread_cap_ticks"],
        )

    # ── Emit per (target, horizon) CSV ───────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bar_tag = f"_bar{args.bar_size}s" if args.bar_size > 0 else ""
    manifest_rows = []

    for (target_net, H), y in labels.items():
        out_df = features.copy()
        out_df["label3"] = y.values

        dist = out_df["label3"].value_counts()
        print(f"\n  t={target_net}, H={H}s: {len(out_df):,} rows")
        print(f"    flat={dist.get(0,0):,}  up={dist.get(1,0):,}  dn={dist.get(2,0):,}")

        fname = f"l2_compact_{args.symbol.lower()}{bar_tag}_t{target_net}_h{H}_{ts}.csv"
        fpath = out_dir / fname
        out_df.to_csv(fpath, index=False)
        manifest_rows.append({
            "file": fname,
            "target_net_ticks": target_net,
            "horizon_s": H,
            "rows": len(out_df),
            "symbol": args.symbol.lower(),
            "bar_size": args.bar_size,
        })
        print(f"    → {fpath}")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(f"\nManifest saved to {out_dir / 'manifest.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
