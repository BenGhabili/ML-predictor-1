# services/orderflow_labels.py
from __future__ import annotations
import numpy as np
import pandas as pd


def _pair_targets_horizons(targets_net_ticks: list, horizons_s: list) -> list[tuple[float, int]]:
    if len(targets_net_ticks) == len(horizons_s):
        return list(zip(targets_net_ticks, horizons_s))
    if len(targets_net_ticks) == 1:
        return [(targets_net_ticks[0], H) for H in horizons_s]
    if len(horizons_s) == 1:
        return [(t, horizons_s[0]) for t in targets_net_ticks]
    raise ValueError("targets_net_ticks and horizons_s lengths are incompatible")


def make_labels_multi(
    df: pd.DataFrame,
    *,
    targets_net_ticks: list,
    horizons_s: list,
    tick_size: float,
    tick_value_usd: float,
    fees_roundtrip_usd: float,
    slippage_ticks_per_side: float,
    entry_delay_ms: int,
    spread_cap_ticks: float,
    price_col: str = "Last",
    spread_col: str = "Spread",
) -> dict[tuple[float, int], pd.Series]:
    prices = df[price_col].to_numpy()
    times = df.index.view("int64")
    spread = df[spread_col].to_numpy() if spread_col in df.columns else np.zeros_like(prices)

    fees_ticks = float(fees_roundtrip_usd) / float(tick_value_usd)
    slip_ticks = 2.0 * float(slippage_ticks_per_side)
    cost_ticks = fees_ticks + slip_ticks

    pairs = _pair_targets_horizons(targets_net_ticks, horizons_s)

    labels = {}
    max_wait_ns = int(max(horizons_s) * 1e9)
    start_delay_ns = int(entry_delay_ms * 1e6)
    spread_limit = spread_cap_ticks * float(tick_size)

    for target_net, H in pairs:
        tp_gross_ticks = float(target_net) + cost_ticks
        delta = tp_gross_ticks * float(tick_size)
        horizon_ns = int(H * 1e9)

        y = np.zeros(len(df), dtype=np.int8)  # 0=flat, 1=down, 2=up

        j_start = 0
        for i in range(len(df)):
            if spread[i] > spread_limit:
                y[i] = 0
                continue

            t0 = times[i] + start_delay_ns
            end_ns = t0 + min(horizon_ns, max_wait_ns)

            if j_start < i + 1:
                j_start = i + 1
            j0 = np.searchsorted(times, t0, side="left")
            j_end = np.searchsorted(times, end_ns, side="right")
            if j0 >= j_end:
                y[i] = 0
                continue

            p0 = prices[i]
            up = p0 + delta
            dn = p0 - delta

            seg = prices[j0:j_end]
            up_mask = seg >= up
            dn_mask = seg <= dn

            up_idx = np.argmax(up_mask) if up_mask.any() else -1
            dn_idx = np.argmax(dn_mask) if dn_mask.any() else -1

            if up_idx == -1 and dn_idx == -1:
                y[i] = 0
            elif up_idx == -1:
                y[i] = 1  # down first
            elif dn_idx == -1:
                y[i] = 2  # up first
            else:
                y[i] = 2 if up_idx < dn_idx else 1

        labels[(target_net, H)] = pd.Series(y, index=df.index, name=f"label3_t{target_net}_h{H}")

    return labels
