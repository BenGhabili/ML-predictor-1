#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import requests
import pandas as pd


def parse_nt8_line(line: str):
    parts = line.strip().split(';')
    if len(parts) < 5:
        return None
    head = parts[0].split()
    if len(head) < 2:
        return None
    dt = pd.to_datetime(f"{head[0]} {head[1]}", format="%Y%m%d %H%M%S", utc=True)
    last = float(parts[1]); bid = float(parts[2]); ask = float(parts[3]); vol = float(parts[4])
    return {
        "ts_utc": int(dt.value // 1_000_000),
        "last": last,
        "bid": bid,
        "ask": ask,
        "size": vol,
    }


def iter_ticks(path: Path):
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            if line.lstrip().startswith('{'):
                # JSONL
                try:
                    yield json.loads(line)
                except Exception:
                    continue
            else:
                snap = parse_nt8_line(line)
                if snap:
                    yield snap


def main():
    p = argparse.ArgumentParser(description="Replay orderflow ticks to FastAPI /predict")
    p.add_argument("--input", required=True)
    p.add_argument("--url", default="http://127.0.0.1:8000/predict")
    p.add_argument("--mode", choices=["realtime", "fixed"], default="realtime")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--batch-ms", type=int, default=25)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--log", default="data/replay_log.csv")
    p.add_argument("--symbol", default="NQ")
    p.add_argument("--contract", default="12-25")
    p.add_argument("--max", type=int, default=0)
    args = p.parse_args()

    target = args.url
    if args.threshold > 0:
        target += f"?threshold={args.threshold}"

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    latencies = []

    last_ts = None
    sent = 0

    for snap in iter_ticks(Path(args.input)):
        snap.setdefault("symbol", args.symbol)
        snap.setdefault("contract", args.contract)
        now_ts = snap["ts_utc"]

        if args.mode == "realtime" and last_ts is not None:
            dt = max(0, int((now_ts - last_ts) / args.speed))
            if dt > 0:
                time.sleep(dt / 1000.0)
        elif args.mode == "fixed":
            time.sleep(max(1, args.batch_ms) / 1000.0)

        t0 = time.perf_counter()
        resp = requests.post(target, json=snap, timeout=2.0)
        t1 = time.perf_counter()

        lat_ms = (t1 - t0) * 1000.0
        latencies.append(lat_ms)

        data = resp.json() if resp.ok else {"error": resp.text}
        out_rows.append({
            "ts_utc": snap.get("ts_utc"),
            "last": snap.get("last"),
            "bid": snap.get("bid"),
            "ask": snap.get("ask"),
            "size": snap.get("size"),
            "p_up": data.get("p_up", 0.0),
            "p_dn": data.get("p_dn", 0.0),
            "score": data.get("score", 0.0),
            "signal": data.get("signal", ""),
            "latency_ms": lat_ms,
        })

        sent += 1
        last_ts = now_ts
        if sent % 100 == 0 and latencies:
            arr = pd.Series(latencies[-100:])
            print(f"last100 latency p50={arr.quantile(0.5):.2f} p95={arr.quantile(0.95):.2f} p99={arr.quantile(0.99):.2f} ms")

        if args.max and sent >= args.max:
            break

    pd.DataFrame(out_rows).to_csv(args.log, index=False)
    print(f"Wrote {args.log} rows={len(out_rows)}")


if __name__ == "__main__":
    main()
