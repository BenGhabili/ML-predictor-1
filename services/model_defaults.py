# services/model_defaults.py
"""Single source of truth for model hyper-parameter defaults."""

RF = dict(
    TREES=600,
    DEPTH=None,
    MIN_LEAF=1,
)

XGB = dict(
    TREES=800,
    DEPTH=7,
    ETA=0.05,
    SUBSAMPLE=0.8,
    COLSAMPLE=0.8,
    SCALE_POS_WEIGHT=1.0,
    EARLY_STOP=50,
)

KNN = dict(
    K=7,
)