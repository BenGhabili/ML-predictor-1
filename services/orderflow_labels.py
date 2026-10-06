# services/orderflow_labels.py
from __future__ import annotations
import numpy as np
import pandas as pd

try:
    from numba import njit

    @njit
    def _label_loop(prices, times, spread, spread_limit, start_delay_ns,
                    horizon_ns, max_wait_ns, delta):
        n = len(prices)
        y = np.zeros(n, dtype=np.int8)
        for i in range(n):
            if spread[i] > spread_limit:
                continue

            t0 = times[i] + start_delay_ns
            end_ns = t0 + min(horizon_ns, max_wait_ns)

            j0 = np.searchsorted(times, t0)
            j_end = np.searchsorted(times, end_ns)
            if j0 >= j_end or j0 >= n:
                continue

            p0 = prices[j0]
            up_thr = p0 + delta
            dn_thr = p0 - delta

            up_idx = -1
            dn_idx = -1
            for k in range(j0, min(j_end, n)):
                if up_idx == -1 and prices[k] >= up_thr:
                    up_idx = k - j0
                if dn_idx == -1 and prices[k] <= dn_thr:
                    dn_idx = k - j0
                if up_idx != -1 and dn_idx != -1:
                    break

            if up_idx == -1 and dn_idx == -1:
                y[i] = 0
            elif up_idx == -1:
                y[i] = 1
            elif dn_idx == -1:
                y[i] = 2
            else:
                y[i] = 2 if up_idx < dn_idx else 1
        return y

except ImportError:

    def _label_loop(prices, times, spread, spread_limit, start_delay_ns,
                    horizon_ns, max_wait_ns, delta):
        n = len(prices)
        y = np.zeros(n, dtype=np.int8)
        for i in range(n):
            if spread[i] > spread_limit:
                continue

            t0 = times[i] + start_delay_ns
            end_ns = t0 + min(horizon_ns, max_wait_ns)

            j0 = int(np.searchsorted(times, t0, side="left"))
            j_end = int(np.searchsorted(times, end_ns, side="right"))
            if j0 >= j_end or j0 >= n:
                continue

            p0 = prices[j0]
            up_thr = p0 + delta
            dn_thr = p0 - delta

            seg = prices[j0:j_end]
            up_mask = seg >= up_thr
            dn_mask = seg <= dn_thr

            up_idx = int(np.argmax(up_mask)) if up_mask.any() else -1
            dn_idx = int(np.argmax(dn_mask)) if dn_mask.any() else -1

            if up_idx == -1 and dn_idx == -1:
                y[i] = 0
            elif up_idx == -1:
                y[i] = 1
            elif dn_idx == -1:
                y[i] = 2
            else:
                y[i] = 2 if up_idx < dn_idx else 1
        return y


def _pair_targets_horizons(targets_net_ticks: list, horizons_s: list) -> list[tuple[float, int]]:
    if len(targets_net_ticks) == len(horizons_s):
        return list(zip(targets_net_ticks, horizons_s))
    if len(targets_net_ticks) == 1:
        return [(targets_net_ticks[0], H) for H in horizons_s]
    if len(horizons_s) == 1:
        return [(t, horizons_s[0]) for t in targets_net_ticks]
    # Different lengths, both > 1: full grid (cross product)
    return [(t, H) for t in targets_net_ticks for H in horizons_s]


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
    prices = df[price_col].to_numpy(dtype=np.float64)
    times = df.index.view("int64")
    spread = df[spread_col].to_numpy(dtype=np.float64) if spread_col in df.columns else np.zeros_like(prices)

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

        y = _label_loop(prices, times, spread, spread_limit,
                        start_delay_ns, horizon_ns, max_wait_ns, delta)

        labels[(target_net, H)] = pd.Series(y, index=df.index, name=f"label3_t{target_net}_h{H}")

    return labels
