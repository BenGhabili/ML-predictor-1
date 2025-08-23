## CURSOR chat history — orderflow v0.21 (concise)

### What we built
- Parallel orderflow pipeline (ticks) alongside minute-bar code (unchanged).
- Data prep → compact features → event labels → CSV per (target,horizon).
- Training with existing trainer via --input CSV; invoke tasks added.
- FastAPI predictor `api/orderflow_app.py` with /predict and /reset.
- Stateful runtime computes the same 12 features used in training.

### Canonical features (order matters)
Spread, last_minus_mid, aggressor, signed_size, cdelta_50, cdelta_100, ti_count_500ms, ret_1s, vol_2s, spread_z_60s, sin_time, cos_time.

### Labels (v0.21)
- label3: 0=flat, 1=down, 2=up; mapping fixed.
- Use net targets; convert to gross by adding fees+slippage; 1:1 tp/sl.
- entry_delay_ms applied to label start, exit_delay_ms unused.
- Spread cap: if spread(t) > cap → label=0.
- Pair targets/horizons by index; broadcast if one list len=1.

### Config and manifests
- Keep manifests in Git:
  - models/orderflow-models.csv (model registry; no .pkl in Git)
  - data_orderflow/manifest.csv (list of produced CSVs; not the big CSVs)
- You’ll copy .pkl and big CSVs manually to other machines.

### Training results (RTH, ~360k rows)
- t1_h10 tuned (trees=1000, eta=0.03, depth=6): CV macro-F1 ≈ 0.634 ± 0.060
- t4_h20 tuned: ≈ 0.487 ± 0.092
- t8_h45 tuned: ≈ 0.458 ± 0.107
- Longer horizons improved t4/t8; t1_h10 best.

### FastAPI predictor
- Path: `api/orderflow_app.py`; env ORDERFLOW_MODEL_PATH selects model.
- Endpoints: GET /ping, POST /reset, POST /predict (v0.21 mapping).
- Score = p_up − p_dn; optional ?threshold returns long/short/flat.
- Warm-up returns `{warmup:true}`; bad quotes/spread return zeros.

### Invoke tasks (orderflow)
- `invoke of-api --model models/xgb_of_t1_h10.pkl --port 8000`
- `invoke of-prep --input data_raw/NQ_06-25_tick.txt --session-filter rth --max-rows 500000 --targets 1,4,8 --horizons 10,20,45`
- `invoke of-train --input data_orderflow/orderflow_compact_t1.0_h10.csv --algo xgb --cv walk [--save]`
- Mock/replay:
  - `invoke of-gen --seconds 60 --bps 40 --trend up --out data/mock_ticks.jsonl`
  - `curl -X POST http://127.0.0.1:8000/reset`
  - `invoke of-replay --input data/mock_ticks.jsonl --threshold 0.20 --log data/replay_log.csv`

### Latency sanity (replay)
- Observed p50 ~9–10 ms, p95 ~11–12 ms, p99 mostly <15 ms on localhost.
- Use persistent client / uvloop for extra margin; 1 worker recommended unless multiple streams.

### Windows notes
- Create `.venv`, `pip install -r requirements.txt`.
- Start API with `py -m invoke of-api --model models\xgb_of_t1_h10.pkl --port 8000`.
- Copy `.pkl` models and any big CSVs manually.

### Next ideas
- Class-weighted XGBoost (no SMOTE); light hyperparam tuning.
- Simple imputation for NaNs (retain rows).
- Offline validator to align /predict outputs with offline labels.


