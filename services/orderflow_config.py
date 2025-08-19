# services/orderflow_config.py
from __future__ import annotations
from pathlib import Path
import copy
import yaml


_DEFAULT_CFG = {
    "general": {
        "symbol": "NQ",
        "tz_exchange": "America/Chicago",
    },
    "sessions": {
        "session_filter": "all",  # all | rth | london
        "rth_hours": "08:30-15:00",
        "london_hours": "02:00-11:00",
        "tz_london": "Europe/London",
    },
    "costs": {
        "tick_size": 0.25,
        "tick_value_usd": 5.0,
        "fees_roundtrip_usd": 4.5,
        "slippage_ticks_per_side": 0.75,
    },
    "delays": {
        "entry_delay_ms": 200,
        "exit_delay_ms": 100,
    },
    "labels": {
        "targets_net_ticks": [1, 4, 8, 12, 20],
        "horizons_s": [5, 10, 30, 45, 90],
        "spread_cap_ticks": 8,
    },
}


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(yaml_path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(_DEFAULT_CFG)

    data = {}
    if yaml_path:
        p = Path(yaml_path)
        if p.exists():
            with p.open() as fh:
                try:
                    data = yaml.safe_load(fh) or {}
                except Exception:
                    data = {}
    _deep_update(cfg, data)
    _deep_update(cfg, overrides or {})
    return cfg
