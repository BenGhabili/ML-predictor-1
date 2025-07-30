# services/model_factory.py
# ── imports ────────────────────────────────────────────────────────────
from services.model_defaults import RF, XGB, KNN
from services.train_utils import (
    train_rf_model_over_sampled,
    train_knn_model,
    build_xgb_model,          # now returns a *fitted* model
)

# ── factory ────────────────────────────────────────────────────────────
def build_model(algo, X, y, explain, **kw):
    if algo == "rf":
        return train_rf_model_over_sampled(
            X, y, n_trees=kw.get("trees", RF["TREES"]))[0]

    if algo == "knn":
        return train_knn_model(X, y, n_neighbors=kw.get("k", KNN["K"]))[0]

    if algo == "xgb":
        return build_xgb_model(
            X, y,
            n_estimators=kw.get("trees", XGB["TREES"]),
            learning_rate=kw.get("eta", XGB["ETA"]),
            max_depth=kw.get("depth", XGB["DEPTH"]),
            X_val=kw.get("X_val"),
            y_val=kw.get("y_val"),
            explain=explain
        )

    raise ValueError(f"Unknown algo: {algo}")
