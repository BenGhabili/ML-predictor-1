# v0.21 Addendum — delta from v0.2 (clarifications only)

- **Label encoding locked:** label3 = {0: flat, 1: down, 2: up}.
- **Spread cap behavior locked:** if spread(t) > spread_cap_ticks[symbol], set label = 0 (flat); do not drop rows.
- **Label delay clarified:** apply entry_delay_ms to label start; exit_delay_ms is not used in labels.
- **Targets & horizons (unchanged):** targets_net_ticks = [5,10,20], horizons_s = [20,45,90]; pair by index (zip). If one list has length 1, broadcast; else config error.
- **I/O (unchanged):** emit CSV per (target,horizon) with compact features + label3; train with existing `modules/train.py --input <csv>`.
- **Scope reaffirmed:** no policy/EV/drawdown/experiment runner in v0.21.
- **Time handling (unchanged):** TXT timestamps treated as UTC; session_filter optional; sin/cos time computed from exchange time.


# v0.2 Addendum — Baseline (CSV, pandas offline)

**Builds on v0.1.** This addendum only clarifies timezones/sessions, label generation, and I/O so the orderflow pipeline plugs into the existing trainer.

## Goal
Create a tick-based **orderflow** dataset from NinjaTrader TXT (Last/Bid/Ask/Volume), add a compact feature set, generate **spread-aware first-touch labels**, and train with the existing `modules/train.py` via `--input <csv>`. No repo reshuffle.

---

## Config (single `cfg` dict; precedence = CLI → API → YAML)
**Time & sessions**
- `timestamps_in: "UTC"`  *(TXT treated as UTC)*
- `tz_exchange: "America/Chicago"`  *(CME RTH = 08:30–15:00 exchange time)*
- `tz_london:   "Europe/London"`     *(handles GMT/BST automatically)*
- `session_filter: "all"`            *(one of: all | rth | london)*
- `rth_hours: "08:30-15:00"`         *(interpreted in tz_exchange)*
- `london_hours: "07:00-17:00"`      *(interpreted in tz_london)*

**Costs & delays**
- `tick_size: {NQ: 0.25, ES: 0.25}`
- `tick_value_usd: {NQ: 5.0, ES: 12.5}`
- `fees_roundtrip_usd: {NQ: 4.5, ES: 5.0}`
- `slippage_ticks_per_side: 0.75`
- `entry_delay_ms: 200`   *(applied in labels)*
- `exit_delay_ms: 100`    *(reserved for later; not used in v0.2 labels)*

**Targets & guards**
- `targets_net_ticks: [1, 4, 8, 12, 20]`  *(“net” = profit after costs)*
- `horizons_s:        [5, 10, 30, 45, 90]` *(paired by index; see rule below)*
- `spread_cap_ticks:  {NQ: 8, ES: 4}`      *(per symbol)*
- `label_encoding: {flat: 0, down: 1, up: 2}`

**Reporting thresholds (for v0.2 summaries)**
- `max_latency_ms: 100`
- `min_expectancy_usd: 0.75`

---

## Session handling
`filter_session(df, cfg)`:
- `"all"`: no filter  
- `"rth"`: convert `ts_utc → tz_exchange`, keep rows within `rth_hours`  
- `"london"`: convert `ts_utc → tz_london`, keep rows within `london_hours`  
Always compute time-of-day sin/cos features (use **exchange time** as base).

---

## Labels (first-touch, spread-aware, **net targets**)
For each timestamp *t* and each `(target_net_ticks, horizon_s)` pair:

1) **Convert net → gross thresholds** *(inside the labeller)*  
   - `fees_ticks = fees_roundtrip_usd / tick_value_usd[symbol]`  
   - `slip_ticks = 2 * slippage_ticks_per_side`  
   - `cost_ticks = fees_ticks + slip_ticks`  
   - `tp_gross = target_net_ticks + cost_ticks`  
   - *(v0.2 assumes 1:1; if you later set an explicit SL target, add `cost_ticks` there too.)*

2) **Start time:** `t0 = t + entry_delay_ms`

3) **Boundaries:**  
   - `Upper = last(t0) + tp_gross * tick_size[symbol]`  
   - `Lower = last(t0) - tp_gross * tick_size[symbol]`  *(1:1 in v0.2)*

4) **Outcome within `horizon_s`:**  
   - Hit Upper first → **up (2)**  
   - Hit Lower first → **down (1)**  
   - Neither → **flat (0)**

5) **Spread cap:** if `spread(t) > spread_cap_ticks[symbol]`, set **flat (0)** (or skip consistently; pick one and stick to it).

**Horizon pairing rule:**  
- If one list has length 1, **broadcast** across the other.  
- If both lists have length >1 and differ → **config error**.

---

## Features (past-only, compact)
Return a DataFrame aligned by timestamp with at least:
- `spread = ask - bid`
- `mid = (bid + ask)/2`
- `last_minus_mid = last - mid`
- `aggressor` *(quote test: +1 at ask, −1 at bid, else 0)*
- `signed_size = aggressor * size`
- `cdelta_50`, `cdelta_100`  *(rolling by trades)*
- `trades_per_500ms`
- `ret_1s` *(points or ticks—be consistent)*
- `vol_2s`
- `spread_z_60s`
- `sin_time`, `cos_time` *(from exchange time)*

---

## I/O (v0.2 = CSV for training)
- Emit one **CSV per (target,horizon)** containing the features above + `label3` (0 flat, 1 down, 2 up).  
- Suggested path: `data/processed_orderflow_{symbol}_{contract}_H{H}s_T{T}net_{ts}.csv`.

---

## How to train (unchanged)
Use the existing trainer:
```
python modules/train.py --algo xgb --input data/processed_orderflow_...csv --cv walk --save
```
## v0.2 “done” check
- At least two CSVs produced (e.g., **T4/H10s** and **T8/H30s**) with the compact feature set + `label3`.  
- `modules/train.py --input <csv>` runs and reports CV results.  
- A brief metrics summary (console or `metrics.csv`) shows **coverage (% non-zero), hit-rate (non-zero), avg win/lose (ticks), net expectancy (USD)** using the same costs/delays.

----------------------- --------------------------- ----------


# Scalping ML — WORKING DRAFT v0.1

**Status:** experimental. Treat all values as hypotheses; change freely. Record changes in the Decision Log below.

## Scope
- Instruments: ES, NQ (futures)
- Data: NinjaTrader tick TXT → normalised Parquet (last, bid, ask, size; UTC)
- Objective: short-horizon direction/pnl edge (2–10s) with strict latency

## Data contract
TXT line: `yyyyMMdd HHmmss fffffff; last; bid; ask; volume` (UTC)  
Validation: sort+dedup by timestamp; drop rows with `bid > ask` or non-positive prices; spread caps by symbol.

## Labels
Event-based: **first ±1 tick** move within `H ∈ {2,5,10}s`.  
Alternative label: cross-spread PnL using fees/slippage constants.

## Features (first pass)
spread, last−mid, quote-test aggressor (±1/0), signed size, rolling cumulative delta (25/50/100 trades), trade intensity (250/500/1000 ms), simple returns/volatility, clock sin/cos.

## Training protocol
Walk-forward: train 20 sessions → validate 5 → slide.  
Model: LightGBM/XGBoost baseline. Importance pruning; keep top 10–12.

## Profitability Gate (must pass to deploy)
- Net expectancy per trade (after **fees+slippage**) ≥ `config.promotion_gate.min_expectancy_usd`
- Latency (feature→signal→order) ≤ `config.promotion_gate.latency_ms_max`
- Stability (KS/AD) p-value ≥ `config.promotion_gate.ks_p_threshold`

## Live rules (hot path)
No orchestrator or HTTP/JSON. Predictor in-process with ring buffers (no pandas). Async logging. Risk limits enforced.

---

## Decision Log (ADR-lite)
- 2025-08-11 — v0.1 baseline created: first ±1 tick label, horizons 2/5/10s; fees+slippage seeded; volume-based roll.
- (add entries here)
