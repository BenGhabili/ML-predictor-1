# api/orderflow_app.py
from __future__ import annotations
import os
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from joblib import load

from services.orderflow_runtime import FeatureState, FeatureStateConfig
from services.orderflow_features import OF_FEATURE_ORDER


class Tick(BaseModel):
    symbol: str = Field(..., examples=["NQ"]) 
    contract: Optional[str] = Field(None, examples=["12-25"]) 
    ts_utc: int = Field(..., description="ms since epoch (UTC)")
    last: float
    bid: float
    ask: float
    size: float = 1


app = FastAPI(title="Orderflow Predictor", version="0.21")

MODEL = None
STATE = None
CFG: Optional[FeatureStateConfig] = None


@app.on_event("startup")
def startup_event():
    global MODEL, STATE, CFG
    model_path = os.environ.get("ORDERFLOW_MODEL_PATH")
    if not model_path:
        # Allow alternative env var name for convenience
        model_path = os.environ.get("OF_MODEL")
    if not model_path or not os.path.exists(model_path):
        raise RuntimeError("ORDERFLOW_MODEL_PATH not set or file not found")

    MODEL = load(model_path)

    CFG = FeatureStateConfig(
        tz_exchange="America/Chicago",
        spread_cap_ticks={"NQ": 8, "ES": 4},
        tick_size_map={"NQ": 0.25, "ES": 0.25},
    )
    STATE = FeatureState(CFG)


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/reset")
def reset_state():
    """Test-only: reset rolling feature state."""
    global STATE
    if CFG is None:
        raise HTTPException(status_code=500, detail="CFG not initialized")
    STATE = FeatureState(CFG)
    return {"status": "reset"}


@app.post("/predict")
def predict(tick: Tick, threshold: Optional[float] = Query(None)):
    if tick.bid > tick.ask:
        return {"ts_utc": tick.ts_utc, "p_up": 0.0, "p_dn": 0.0, "score": 0.0, "reason": "bad_quotes"}

    df = STATE.update(tick.model_dump())
    if df is None:
        return {"ts_utc": tick.ts_utc, "p_up": 0.0, "p_dn": 0.0, "score": 0.0, "warmup": True}

    try:
        X = df[OF_FEATURE_ORDER].to_numpy()
        probs = MODEL.predict_proba(X)[0]
        # label mapping fixed: index 0=flat, 1=down, 2=up
        p_flat, p_dn, p_up = float(probs[0]), float(probs[1]), float(probs[2])
        score = p_up - p_dn
        resp = {"ts_utc": tick.ts_utc, "p_up": p_up, "p_dn": p_dn, "score": score}
        if threshold is not None:
            sig = "flat"
            if score >= threshold:
                sig = "long"
            elif score <= -threshold:
                sig = "short"
            resp["signal"] = sig
        return resp
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

