# modules/orderflow_data_prep.py
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

from services.orderflow_config import load_config
from services.orderflow_data_utils import load_orderflow_ticks, _infer_symbol_and_contract_from_filename
from services.orderflow_features import build_orderflow_features, select_compact_features
from services.orderflow_labels import make_labels_multi
from services.orderflow_sessions import filter_session


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_list_numbers(s: str | None, cast=float):
    if not s:
        return None
    return [cast(x.strip()) for x in s.split(',') if x.strip()]


def main(*, input_path: Path, out_dir: Path = Path("data_orderflow"), overrides: dict | None = None, no_save: bool = False) -> None:
    cfg = load_config("config.yaml", overrides or {})

    print(f"Loading ticks from: {input_path}")
    df = load_orderflow_ticks(input_path, max_rows=overrides.get("max_rows") if overrides else None)
    print(f"Loaded {len(df):,} ticks")

    # Session filter
    sess = cfg.get("sessions", {})
    df_sess = filter_session(
        df,
        session_filter=sess.get("session_filter", "all"),
        tz_exchange=cfg["general"].get("tz_exchange", "America/Chicago"),
        rth_hours=sess.get("rth_hours", "08:30-15:00"),
        tz_london=sess.get("tz_london", "Europe/London"),
        london_hours=sess.get("london_hours", "02:00-11:00"),
    )
    print(f"After session filter: {len(df_sess):,} rows")

    # Features
    df_feat = build_orderflow_features(
        df_sess,
        windows_trades=(25, 50, 100),
        windows_ms=(250, 500, 1000),
        exchange_tz=cfg["general"].get("tz_exchange", "America/Chicago"),
    )

    # Labels (multi)
    costs = cfg.get("costs", {})
    delays = cfg.get("delays", {})
    labels_cfg = cfg.get("labels", {})
    labels = make_labels_multi(
        df_feat,
        targets_net_ticks=labels_cfg.get("targets_net_ticks", [1, 4, 8, 12, 20]),
        horizons_s=labels_cfg.get("horizons_s", [5, 10, 30, 45, 90]),
        tick_size=costs.get("tick_size", 0.25),
        tick_value_usd=costs.get("tick_value_usd", 5.0),
        fees_roundtrip_usd=costs.get("fees_roundtrip_usd", 4.5),
        slippage_ticks_per_side=costs.get("slippage_ticks_per_side", 0.75),
        entry_delay_ms=delays.get("entry_delay_ms", 200),
        spread_cap_ticks=labels_cfg.get("spread_cap_ticks", 8),
    )

    # Determine symbol/contract tags for filenames
    sym_tag = None
    contract_tag = None

    try:
        if "symbol_root" in df_feat.columns:
            s = df_feat["symbol_root"].dropna()
            if len(s) > 0:
                sym_tag = str(s.iloc[0]).lower()
        if "contract" in df_feat.columns:
            c = df_feat["contract"].dropna()
            if len(c) > 0:
                contract_tag = str(c.iloc[0]).lower()
    except Exception:
        pass

    if sym_tag is None or contract_tag is None:
        try:
            sym_i, con_i = _infer_symbol_and_contract_from_filename(input_path)
            if sym_tag is None and sym_i:
                sym_tag = sym_i.lower()
            if contract_tag is None and con_i:
                contract_tag = con_i.lower()
        except Exception:
            pass

    sym_tag = sym_tag or "na"
    contract_tag = contract_tag or "na"

    # Emit per (target,horizon) compact CSV
    compact = select_compact_features(df_feat)
    ensure_dir(out_dir)
    manifest_rows = []

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for (target_net, H), y in labels.items():
        # Always use 'label3' to match trainer expectations
        out_df = compact.copy()
        out_df['label3'] = y.values

        if no_save:
            print(f"Sample for t={target_net}, H={H}s:")
            print(out_df.head())
            continue

        fname = f"orderflow_compact_{sym_tag}_{contract_tag}_t{target_net}_h{H}_{ts}.csv"
        fpath = out_dir / fname
        out_df.to_csv(fpath, index=False)
        manifest_rows.append({
            "file": fname,
            "target_net_ticks": target_net,
            "horizon_s": H,
            "rows": len(out_df),
            "symbol": sym_tag,
            "contract": contract_tag,
        })
        print(f"Wrote {fpath}")

    if not no_save:
        manifest = pd.DataFrame(manifest_rows)
        manifest.to_csv(out_dir / "manifest.csv", index=False)
        print("Manifest saved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Orderflow data prep v0.2 — emit compact CSVs per (target,horizon)")
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", default="data_orderflow")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--session_filter", default=None)
    p.add_argument("--tz_exchange", default=None)
    p.add_argument("--rth_hours", default=None)
    p.add_argument("--london_hours", default=None)
    p.add_argument("--targets_net_ticks", default=None, help="Comma list, e.g. 1,4,8,12,20")
    p.add_argument("--horizons_s", default=None, help="Comma list, e.g. 5,10,30,45,90")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    overrides = {"max_rows": args.max_rows}
    if args.session_filter:
        overrides.setdefault("sessions", {})["session_filter"] = args.session_filter
    if args.tz_exchange:
        overrides.setdefault("general", {})["tz_exchange"] = args.tz_exchange
    if args.rth_hours:
        overrides.setdefault("sessions", {})["rth_hours"] = args.rth_hours
    if args.london_hours:
        overrides.setdefault("sessions", {})["london_hours"] = args.london_hours
    if args.targets_net_ticks:
        overrides.setdefault("labels", {})["targets_net_ticks"] = _parse_list_numbers(args.targets_net_ticks, cast=float)
    if args.horizons_s:
        overrides.setdefault("labels", {})["horizons_s"] = _parse_list_numbers(args.horizons_s, cast=int)

    main(input_path=Path(args.input), out_dir=Path(args.out_dir), overrides=overrides, no_save=args.no_save)
