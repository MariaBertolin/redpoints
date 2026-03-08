import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)


def metric_kwargs(y_true):
    """Kwargs seguros para precision/recall/f1 (binario {0,1} vs resto)."""
    y_true = np.asarray(y_true)
    uniq = np.unique(y_true)

    if len(uniq) == 2 and set(uniq.tolist()) == {0, 1}:
        return {"average": "binary", "pos_label": 1, "zero_division": 0}

    return {"average": "macro", "zero_division": 0}


def compute_metrics(y_true, y_pred, y_proba=None):
    """
    Métricas esenciales.
    - loss (log_loss) solo si se pasa y_proba (p.ej. predict_proba).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    kw = metric_kwargs(y_true)

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, **kw),
        "recall": recall_score(y_true, y_pred, **kw),
        "f1": f1_score(y_true, y_pred, **kw),
    }

    if y_proba is not None:
        try:
            out["loss"] = log_loss(y_true, y_proba)
        except Exception:
            out["loss"] = None  # mantiene el script simple (sin reventar)
    else:
        out["loss"] = None

    return out
# Estaria guay posar algo com un PCA, més gràfic, però això és enough
def evaluate_thresholds(y_true, y_proba, thresholds=None):
    if thresholds is None:
        thresholds = [i / 100 for i in range(10, 91, 5)]  # 0.10 a 0.90, he vist que 0.2 és el que més prioritza el recall sense tenir mals valors de les altres mètriques

    rows = []

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)

        rows.append({
            "threshold": thr,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })

    return pd.DataFrame(rows)