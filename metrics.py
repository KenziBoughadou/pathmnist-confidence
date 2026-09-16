"""Probabilités, calibration et sélection par confiance, sans entraînement."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import f1_score


def validate_scores(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if logits.ndim != 2 or len(logits) == 0 or labels.shape != (len(logits),):
        raise ValueError("Logits (N, K) et étiquettes (N,) non vides attendus")
    if not np.isfinite(logits).all():
        raise ValueError("Logits non finis")
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("Étiquettes hors des classes")
    return logits, labels


def nll(logits, labels):
    logits, labels = validate_scores(logits, labels)
    return float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels]))


def probabilities(logits, temperature=1.0):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Température strictement positive et finie attendue")
    return softmax(np.asarray(logits, dtype=np.float64) / temperature, axis=1)


def fit_temperature(logits, labels):
    logits, labels = validate_scores(logits, labels)
    # Optimiser log(T) impose T > 0, sans modifier les poids du CNN.
    result = minimize_scalar(
        lambda log_t: nll(logits / np.exp(log_t), labels),
        method="bounded", bounds=(-4.0, 4.0), options={"xatol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Échec de calibration : {result.message}")
    baseline = nll(logits, labels)
    improved = float(result.fun) < baseline
    temperature = float(np.exp(result.x)) if improved else 1.0
    return {
        "temperature": temperature,
        "calibration_nll_raw": baseline,
        "calibration_nll_scaled": nll(logits / temperature, labels),
        "optimizer_success": bool(result.success),
        "optimizer_evaluations": int(result.nfev),
        "log_temperature_bounds": [-4.0, 4.0],
        "near_bound": bool(abs(result.x) > 3.99),
        "kept_identity": not improved,
    }


def reliability_bins(probs, labels, bins=15):
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    # Intervalles [gauche, droite), dernier intervalle fermé à droite.
    assignments = np.minimum((confidence * bins).astype(int), bins - 1)
    rows = []
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        rows.append({
            "lower": index / bins, "upper": (index + 1) / bins,
            "count": count,
            "confidence": float(confidence[mask].mean()) if count else None,
            "accuracy": float(correct[mask].mean()) if count else None,
        })
    return rows


def classification_metrics(logits, labels, temperature=1.0, bins=15):
    logits, labels = validate_scores(logits, labels)
    probs = probabilities(logits, temperature)
    predictions = probs.argmax(axis=1)
    one_hot = np.eye(logits.shape[1])[labels]
    reliability = reliability_bins(probs, labels, bins)
    ece = sum(
        row["count"] / len(labels) * abs(row["accuracy"] - row["confidence"])
        for row in reliability if row["count"]
    )
    return {
        "examples": len(labels),
        "accuracy": float(np.mean(predictions == labels)),
        "macro_f1": float(f1_score(labels, predictions, labels=np.arange(logits.shape[1]), average="macro", zero_division=0)),
        "nll": nll(logits / temperature, labels),
        "brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "ece": float(ece),
        "mean_confidence": float(probs.max(axis=1).mean()),
    }


def risk_coverage(probs, labels):
    confidence = probs.max(axis=1)
    # Les indices d'origine départagent les ex aequo, sans utiliser les étiquettes.
    order = np.argsort(-confidence, kind="stable")
    errors = (probs.argmax(axis=1) != labels)[order]
    accepted = np.arange(1, len(labels) + 1)
    risk = np.cumsum(errors) / accepted
    return {
        "order": order,
        "coverage": accepted / len(labels),
        "risk": risk,
        "aurc": float(risk.mean()),
    }


def threshold_metrics(probs, labels, threshold):
    if not 0 <= threshold <= 1:
        raise ValueError("Seuil hors de [0, 1]")
    mask = probs.max(axis=1) >= threshold
    count = int(mask.sum())
    accuracy = float(np.mean(probs[mask].argmax(axis=1) == labels[mask])) if count else None
    return {
        "threshold": float(threshold), "accepted": count,
        "coverage": count / len(labels), "accuracy": accuracy,
        "risk": None if accuracy is None else 1 - accuracy,
    }


def top_errors(probs, labels, limit=10):
    errors = np.flatnonzero(probs.argmax(axis=1) != labels)
    order = np.argsort(-probs[errors].max(axis=1), kind="stable")
    return errors[order[:limit]]
