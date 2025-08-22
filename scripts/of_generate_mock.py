#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path


def clip_to_grid(x: float, tick: float) -> float:
    return round(round(x / tick) * tick, 10)


def gen_ticks(symbol: str, contract: str, seconds: int, base: float, tick_size: float,
              bps: int, trend: str, spread_ticks: float, seed: int):
    rnd = random.Random(seed)
    ts_ms = int(time.time() * 1000)
    dt_ms = max(1, int(1000 / max(1, bps)))

    last = base
    drift = tick_size if trend == "up" else (-tick_size if trend == "down" else 0.0)

    for _ in range(max(1, seconds) * bps):
        # micro move
        noise = rnd.choice([-tick_size, 0.0, tick_size])
        if trend == "chop":
            move = noise
        else:
            move = drift + noise * 0.5
        last = clip_to_grid(last + move, tick_size)

        # spread with small variation
        spr_ticks = max(1.0, spread_ticks + rnd.uniform(-0.25, 0.25))
        spr = spr_ticks * tick_size
        bid = last - spr / 2.0
        ask = last + spr / 2.0
        if bid > ask:
            bid, ask = ask, bid

        size = rnd.choice([1, 1, 1, 2, 3])

        yield {
            "symbol": symbol,
            "contract": contract,
            "ts_utc": ts_ms,
            "last": float(last),
            "bid": float(bid),
            "ask": float(ask),
            "size": float(size),
        }
        ts_ms += dt_ms


def main():
    p = argparse.ArgumentParser(description="Generate mock orderflow tick snapshots (JSONL)")
    p.add_argument("--symbol", default="NQ")
    p.add_argument("--contract", default="12-25")
    p.add_argument("--seconds", type=int, default=60)
    p.add_argument("--base", type=float, default=17650.0)
    p.add_argument("--tick-size", type=float, default=0.25)
    p.add_argument("--bps", type=int, default=40, help="ticks per second")
    p.add_argument("--trend", choices=["up", "down", "chop"], default="up")
    p.add_argument("--spread", type=float, default=0.5, help="baseline spread in ticks")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/mock_ticks.jsonl")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for tick in gen_ticks(
            symbol=args.symbol,
            contract=args.contract,
            seconds=args.seconds,
            base=args.base,
            tick_size=args.tick_size,
            bps=args.bps,
            trend=args.trend,
            spread_ticks=args.spread,
            seed=args.seed,
        ):
            json.dump(tick, f)
            f.write("\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
