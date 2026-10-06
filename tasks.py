# tasks.py in your project root
from pathlib import Path
import sys
from invoke import task
from datetime import datetime
from shlex import quote

RAW_DIR = Path("data_raw")

@task
def data(ctx):
    ctx.run("python modules/data_preparation_x_min.py --raw-file data_raw/ --timeframe 1 --atr-mult 1.0")

@task
def prepare_minute_data(ctx, timeframe=1, atr=1.0):
    ctx.run(f"python modules/data_preparation_x_min.py --timeframe {timeframe} --atr-mult {atr}")

@task
def prep(ctx,
         timeframe=1,
         atr_mult=1.0,
         from_bars=False,
         prefix="",
         raw_file=None,
         save=False):
    """
    Prepare bar data from a raw price file.

    Examples
    --------
    # Pick interactively from *all* files
    invoke prep --timeframe 1 --atr-mult 0.6

    # Restrict list to files whose names start with 'current_'
    invoke prep --prefix current_

    # Skip the menu and process a known file that is already bars
    invoke prep --raw-file nq_6-25contract.last.txt --from-bars
    """
    # 1) Resolve which raw file to feed the script
    if raw_file is None:
        files = sorted(
            RAW_DIR.glob(f"{prefix}*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if not files:
            print(f"No files in {RAW_DIR} matching prefix '{prefix}'")
            return

        # show the newest 10
        for i, p in enumerate(files[:10]):
            ts = datetime.fromtimestamp(p.stat().st_mtime)     # ← convert
            print(f"[{i}] {p.name}  ({ts:%d-%b-%y %H:%M})")    # now OK

        choice = input("Which file index? [0]: ").strip() or "0"
        raw_file = files[int(choice)].name
    else:
        # user gave a filename
        if not (RAW_DIR / raw_file).exists():
            print(f"{raw_file} not found in {RAW_DIR}")
            return
    save_flag = "--save" if save else ""    
    raw_path = raw_file
    # 2) Build CLI for the prep script
    cmd = (
        f"python modules/data_preparation_x_min.py "
        f"--timeframe {timeframe} "
        f"--atr-mult {atr_mult} "
        f"--raw-file {quote(str(raw_path))} "
        f"{'--from-bars' if from_bars else ''} {save_flag}"
    )
    print("→", cmd)                # plain ASCII arrow
    ctx.run(cmd)
    
    
@task
def simple_train(ctx, timeframe=1):
    """
    Train a KNN model on the most recent prepared CSV for the given timeframe.
    """
    ctx.run(f"python modules/train.py --timeframe {timeframe}")

@task
def train(ctx, timeframe=1, algo="rf", trees=400, depth=10, eta=0.05, k=7, cv="walk", save=False, explain=False):
    # 1) Read the data‑list.csv here
    import pandas as pd
    from pathlib import Path
    df = pd.read_csv(Path("data/data-list.csv"))
    tf_str = f"{timeframe}min"
    df_tf = df[df['Timeframe'] == tf_str].sort_values('Created_At', ascending=False)
    if df_tf.empty:
        print(f"No files for {tf_str}")
        return

    # 2) Show top 3
    for i, row in enumerate(df_tf.head(3).itertuples()):
        print(f"[{i}] {row.Filename}  ({row.Created_At})")

    # 3) Prompt for choice
    # choice = ctx.prompt("Which file index?", default="0")
    choice = input("Which file index? [0]: ") or "0"
    idx = int(choice)
    chosen = df_tf.iloc[idx].Filename

    save_flag = "--save" if save else ""
    explain_flag = "--explain" if explain else ""

    cmd = (
        f"python modules/train.py "
        f"--algo {algo} --timeframe {timeframe} "
        f"--input data/{chosen} --trees {trees} --depth {depth} "
        f"--eta {eta} --k {k} --cv {cv} {save_flag} {explain_flag}"
    )
    ctx.run(cmd.strip())

@task
def api(ctx):
    ctx.run("python api/app.py")

# ---------------- Orderflow (v0.2) tasks -----------------

@task
def of_prep(ctx, input, out_dir="data_orderflow", max_rows=None, targets="1,4,8,12,20", horizons="5,10,30,45,90", session_filter=None, tz_exchange=None, rth_hours=None, london_hours=None, tick_value_usd=None, fees_roundtrip=None, slippage_ticks=None, no_save=False):
    """Orderflow data prep: emit compact CSVs per (target,horizon)."""
    args = ["-m", "modules.orderflow_data_prep", "--input", quote(str(input)), "--out-dir", quote(str(out_dir))]
    if max_rows is not None:
        args += ["--max-rows", str(max_rows)]
    if session_filter:
        args += ["--session_filter", session_filter]
    if tz_exchange:
        args += ["--tz_exchange", tz_exchange]
    if rth_hours:
        args += ["--rth_hours", quote(str(rth_hours))]
    if london_hours:
        args += ["--london_hours", quote(str(london_hours))]
    if targets:
        args += ["--targets_net_ticks", targets]
    if horizons:
        args += ["--horizons_s", horizons]
    if tick_value_usd is not None:
        args += ["--tick-value-usd", str(tick_value_usd)]
    if fees_roundtrip is not None:
        args += ["--fees-roundtrip", str(fees_roundtrip)]
    if slippage_ticks is not None:
        args += ["--slippage-ticks", str(slippage_ticks)]
    if no_save:
        args += ["--no-save"]
    cmd = "python " + " ".join(args)
    ctx.run(cmd)

@task
def of_train(ctx, input, algo="xgb", cv="walk", trees=400, depth=6, eta=0.05, k=7, save=False, explain=False):
    """Train on an orderflow compact CSV (no model registry)."""
    save_flag = "--save" if save else ""
    explain_flag = "--explain" if explain else ""
    cmd = (
        f"python -m modules.train --algo {algo} --input {quote(str(input))} "
        f"--cv {cv} --trees {trees} --depth {depth} --eta {eta} --k {k} {save_flag} {explain_flag}"
    )
    ctx.run(cmd.strip())

@task
def of_api(ctx, host="127.0.0.1", port=8001, model="models/xgb_of_t1_h10.pkl"):
    """Run orderflow FastAPI on a separate port."""
    cmd = f"{sys.executable} -m uvicorn api.orderflow_app:app --host {host} --port {port} --workers 1"
    # Cross-platform: pass model path via env rather than inline assignment
    ctx.run(cmd, env={"ORDERFLOW_MODEL_PATH": str(model)})

@task
def of_sweep(ctx, data_dir="data_orderflow", algo="xgb", cv="walk",
             trees=800, depth=7, eta=0.03, symbol="NQ", save_models=False,
             max_rows=None):
    """Sweep all (target, horizon) combos and rank by profitability."""
    save_flag = "--save-models" if save_models else ""
    rows_flag = f"--max-rows {max_rows}" if max_rows else ""
    cmd = (f"python -m modules.orderflow_sweep "
           f"--data-dir {data_dir} --algo {algo} --cv {cv} "
           f"--trees {trees} --depth {depth} --eta {eta} "
           f"--symbol {symbol} {save_flag} {rows_flag}")
    ctx.run(cmd)

# ---------------- L2 (depth) tasks -----------------

@task
def l2_prep(ctx, input_dir=None, from_cache=None, out_dir="data_l2",
            targets="2,4,8", horizons="10,20,30", symbol="NQ",
            session_filter="08:30-15:00", bar_size=0, grid=False):
    """L2 data prep: parse depth data or use cache → compact feature CSVs."""
    parts = ["python -m modules.l2_data_prep"]
    if from_cache:
        parts += [f"--from-cache {quote(str(from_cache))}"]
    elif input_dir:
        parts += [f"--input-dir {quote(str(input_dir))}"]
    else:
        raise ValueError("Must specify --input-dir or --from-cache")
    parts += [f"--out-dir {quote(str(out_dir))}"]
    parts += [f"--targets {targets}", f"--horizons {horizons}"]
    parts += [f"--symbol {symbol}"]
    if session_filter:
        parts += [f"--session-filter {session_filter}"]
    if bar_size > 0:
        parts += [f"--bar-size {bar_size}"]
    if grid:
        parts += ["--grid"]
    ctx.run(" ".join(parts))


@task
def of_gen(ctx, seconds=60, base=17650.0, bps=40, trend='up', spread=0.5, out='data/mock_ticks.jsonl', symbol='NQ', contract='12-25'):
    ctx.run(f"python scripts/of_generate_mock.py --symbol {symbol} --contract {contract} --seconds {seconds} --base {base} --bps {bps} --trend {trend} --spread {spread} --out {quote(str(out))}")

@task
def of_replay(ctx, input='data/mock_ticks.jsonl', url='http://127.0.0.1:8001/predict', mode='realtime', speed=1.0, batch_ms=25, threshold=0.0, log='data/replay_log.csv'):
    th = f" --threshold {threshold}" if float(threshold) > 0 else ""
    ctx.run(f"python scripts/of_replay.py --input {quote(str(input))} --url {url} --mode {mode} --speed {speed} --batch-ms {batch_ms}{th} --log {quote(str(log))}")