# services/orderflow_sessions.py
from __future__ import annotations
import pandas as pd


def _parse_hours(hours: str) -> tuple[str, str]:
    start, end = hours.split("-")
    return start.strip(), end.strip()


def filter_session(df: pd.DataFrame, *, session_filter: str, tz_exchange: str, rth_hours: str, tz_london: str, london_hours: str) -> pd.DataFrame:
    if session_filter == "all":
        return df

    if session_filter == "rth":
        idx_ex = df.index.tz_convert(tz_exchange)
        start, end = _parse_hours(rth_hours)
        mask = (idx_ex.time >= pd.to_datetime(start).time()) & (idx_ex.time <= pd.to_datetime(end).time())
        return df.loc[mask]

    if session_filter == "london":
        idx_ln = df.index.tz_convert(tz_london)
        start, end = _parse_hours(london_hours)
        mask = (idx_ln.time >= pd.to_datetime(start).time()) & (idx_ln.time <= pd.to_datetime(end).time())
        return df.loc[mask]

    return df
