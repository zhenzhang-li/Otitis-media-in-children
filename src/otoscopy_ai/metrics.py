"""Generic multiclass endpoints; optional named clinical direction is configuration."""

import numpy as np
from scipy.special import softmax
from scipy.stats import norm, t
from sklearn.metrics import confusion_matrix, roc_auc_score


def from_confusion(cm):
    cm = np.asarray(cm, float)
    n = cm.sum((-2, -1))
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    rr = cm.sum(-1)
    cc = cm.sum(-2)
    pr = np.divide(tp, cc, out=np.zeros_like(tp), where=cc > 0)
    rec = np.divide(tp, rr, out=np.zeros_like(tp), where=rr > 0)
    f1 = np.divide(2 * tp, rr + cc, out=np.zeros_like(tp), where=rr + cc > 0)
    tn = n[..., None] - rr - cc + tp
    spec = np.divide(tn, n[..., None] - rr, out=np.zeros_like(tp), where=n[..., None] - rr > 0)
    return {
        "accuracy": tp.sum(-1) / n,
        "macro_precision": pr.mean(-1),
        "macro_recall": rec.mean(-1),
        "macro_f1": f1.mean(-1),
        "macro_specificity": spec.mean(-1),
    }


def wilson(k, n):
    if n == 0:
        return [None, None]
    z = norm.ppf(0.975)
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(center - half), float(center + half)]


def evaluate(y, logits, classes, temperature=1.0, bootstrap=0, seed=0, clinical_pair=None):
    y = np.asarray(y, int)
    logits = np.asarray(logits, float)
    prob = softmax(logits / temperature, axis=1)
    pred = prob.argmax(1)
    cm = confusion_matrix(y, pred, labels=np.arange(len(classes)))
    q = {k: float(v) for k, v in from_confusion(cm).items()}
    q.update(
        n=len(y),
        correct=int((pred == y).sum()),
        confusion=cm.tolist(),
        accuracy_ci=wilson(int((pred == y).sum()), len(y)),
    )
    q["per_class"] = {}
    for i, cl in enumerate(classes):
        n = int(cm[i].sum())
        correct = int(cm[i, i])
        q["per_class"][cl] = {
            "n": n,
            "correct": correct,
            "sensitivity": correct / n if n else None,
            "sensitivity_ci": wilson(correct, n),
            "auc": (
                float(roc_auc_score(y == i, prob[:, i])) if 0 < (y == i).sum() < len(y) else None
            ),
        }
    if clinical_pair:
        a, b = [classes.index(v) for v in clinical_pair]
        q["clinical_pair"] = {
            "labels": clinical_pair,
            "forward_error": float(cm[a, b] / cm[a].sum()) if cm[a].sum() else None,
            "reverse_error": float(cm[b, a] / cm[b].sum()) if cm[b].sum() else None,
            "forward_error_ci": wilson(int(cm[a, b]), int(cm[a].sum())),
            "reverse_error_ci": wilson(int(cm[b, a]), int(cm[b].sum())),
        }
    q["nll"] = float(-np.log(np.maximum(prob[np.arange(len(y)), y], 1e-15)).mean())
    q["brier"] = float(np.square(prob - np.eye(len(classes))[y]).sum(1).mean())
    if bootstrap:
        draws = (
            np.random.default_rng(seed)
            .multinomial(len(y), cm.ravel() / len(y), size=bootstrap)
            .reshape(-1, len(classes), len(classes))
        )
        stat = from_confusion(draws)
        for key in ["macro_precision", "macro_recall", "macro_f1", "macro_specificity"]:
            q[key + "_ci"] = np.quantile(stat[key], [0.025, 0.975]).tolist()
    return q


def mean_ci(values):
    values = np.asarray(values, float)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else None
    half = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else None
    return {
        "n_outer_splits": n,
        "mean": mean,
        "sd": sd,
        "ci_low": mean - half if n > 1 else None,
        "ci_high": mean + half if n > 1 else None,
    }
