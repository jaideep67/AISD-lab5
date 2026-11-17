"""
G5 — Explainable AI Pipeline for CyberSentinel AI
Team 10 · Jaideep (G5 track) · 18 Nov 2026

Research grounding:
  - Mahmoud et al. 2025 (XI2S-IDS) — 2-stage binary→multiclass hierarchy
  - Mohale & Obagbuwa 2025        — TreeSHAP global + local XAI

Architecture:
  Stage 1 (binary XGBoost)   : normal (0) vs attack (1) — fast triage
  Stage 2 (multiclass XGBoost): bruteforce / scan / synflood — attack type
  SHAP Engine (TreeExplainer) : top-N feature attributions per flow

Public API
----------
  pipeline = train_pipeline(Xtr, ytr)
  preds    = predict_pipeline(pipeline, Xte)
  report   = explain_flow(pipeline, flow_vector)
  metrics  = evaluate(pipeline, Xte, yte, feature_names)
"""

from __future__ import annotations
import os, json, time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import shap
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
SEED          = 42
TOP_N_FEATS   = 5          # number of top-|SHAP| features to surface per flow
NORMAL_LABEL  = "normal"
ATTACK_LABELS = ["bruteforce", "scan", "synflood"]   # Stage-2 targets

# ──────────────────────────────────────────────────────────────
# Pipeline container
# ──────────────────────────────────────────────────────────────

@dataclass
class G5Pipeline:
    """Holds both stages, their encoders, and the SHAP explainer."""
    stage1:         XGBClassifier                # binary:  0=normal, 1=attack
    stage2:         XGBClassifier                # 3-class: bruteforce/scan/synflood
    stage1_classes: List[int]                    # [0, 1] — stored separately (read-only prop in xgb)
    stage2_classes: List[str]                    # ordered label list for stage2
    explainer:      shap.TreeExplainer           # pre-built on stage2
    feature_names:  List[str]                    # the 66 feature names (in order)


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def train_pipeline(Xtr: np.ndarray, ytr: np.ndarray,
                   feature_names: List[str] | None = None) -> G5Pipeline:
    """
    Train the two-stage XAI pipeline on training data.

    Parameters
    ----------
    Xtr : (n_train, 66) float array — training features
    ytr : (n_train,)    string array — labels (normal/bruteforce/scan/synflood)
    feature_names : list of 66 names (optional, for SHAP plots)

    Returns
    -------
    G5Pipeline — fitted pipeline with SHAP explainer attached
    """
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(Xtr.shape[1])]

    # ── Stage 1: binary — normal=0, attack=1 ──────────────────
    y1 = (ytr != NORMAL_LABEL).astype(int)   # 0 = normal, 1 = attack
    sw1 = compute_sample_weight("balanced", y1)

    s1 = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        tree_method="hist", n_jobs=1, verbosity=0, random_state=SEED,
        eval_metric="logloss",
    )
    s1.fit(Xtr, y1, sample_weight=sw1)
    # Note: XGBClassifier.classes_ is read-only in xgb>=2 — stored in pipeline dataclass

    # ── Stage 2: multiclass on ATTACK flows only ───────────────
    attack_mask = ytr != NORMAL_LABEL
    Xtr_a = Xtr[attack_mask]
    ytr_a = ytr[attack_mask]                  # bruteforce / scan / synflood

    # Encode string → int for XGBoost
    s2_classes = sorted(set(ytr_a))           # alphabetical: bruteforce, scan, synflood
    enc = {c: i for i, c in enumerate(s2_classes)}
    y2  = np.array([enc[c] for c in ytr_a])
    sw2 = compute_sample_weight("balanced", y2)

    s2 = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        tree_method="hist", n_jobs=1, verbosity=0, random_state=SEED,
        eval_metric="mlogloss",
    )
    s2.fit(Xtr_a, y2, sample_weight=sw2)
    # classes_ is read-only in xgb>=2; stage2_classes stored in G5Pipeline dataclass

    # ── SHAP explainer on Stage 2 ─────────────────────────────
    # TreeExplainer works directly on the booster — no data argument needed
    explainer = shap.TreeExplainer(s2)

    return G5Pipeline(
        stage1=s1,
        stage2=s2,
        stage1_classes=[0, 1],                   # binary: 0=normal, 1=attack
        stage2_classes=s2_classes,
        explainer=explainer,
        feature_names=list(feature_names),
    )


# ──────────────────────────────────────────────────────────────
# Prediction (batch)
# ──────────────────────────────────────────────────────────────

def predict_pipeline(pipeline: G5Pipeline, X: np.ndarray) -> np.ndarray:
    """
    Run the two-stage prediction on a batch of flows.

    Returns
    -------
    preds : (n,) array of string labels
            (normal / bruteforce / scan / synflood)
    """
    n = X.shape[0]
    preds = np.full(n, NORMAL_LABEL, dtype=object)

    # Stage 1: flag attacks
    s1_preds = pipeline.stage1.predict(X)     # 0=normal, 1=attack
    attack_idx = np.where(s1_preds == 1)[0]

    if attack_idx.size > 0:
        X_attack = X[attack_idx]
        s2_int   = pipeline.stage2.predict(X_attack)
        s2_labels = np.array([pipeline.stage2_classes[i] for i in s2_int])
        preds[attack_idx] = s2_labels

    return preds


# ──────────────────────────────────────────────────────────────
# Single-flow explanation (TreeSHAP)
# ──────────────────────────────────────────────────────────────

def explain_flow(pipeline: G5Pipeline, flow: np.ndarray,
                 top_n: int = TOP_N_FEATS) -> dict:
    """
    Explain a single network flow prediction using TreeSHAP.

    Parameters
    ----------
    pipeline   : fitted G5Pipeline
    flow       : (66,) float array — one flow's feature vector
    top_n      : number of top |SHAP| features to return

    Returns
    -------
    dict with keys:
      predicted_class : str  — final prediction (all 4 labels)
      stage1_decision : str  — 'normal' or 'attack'
      top_features    : list of (feature_name, shap_value) tuples, top_n entries
      soc_narrative   : str  — plain-English explanation for a SOC analyst
    """
    x = flow.reshape(1, -1)

    # Stage 1 decision
    s1 = pipeline.stage1.predict(x)[0]
    stage1_label = "normal" if s1 == 0 else "attack"

    if s1 == 0:
        # Normal — no Stage 2 / SHAP needed
        return {
            "predicted_class":  "normal",
            "stage1_decision":  "normal",
            "top_features":     [],
            "soc_narrative":    "Flow classified as NORMAL by Stage-1 filter. No further triage needed.",
        }

    # Stage 2: attack type
    s2_int   = pipeline.stage2.predict(x)[0]
    pred_cls = pipeline.stage2_classes[s2_int]

    # TreeSHAP — shap_values shape: (1, n_features, n_classes)
    shap_vals = pipeline.explainer.shap_values(x)       # (1, 66, 3)
    class_idx = pipeline.stage2_classes.index(pred_cls)
    sv_for_class = shap_vals[0, :, class_idx]           # (66,) for the predicted class

    # Top features by |SHAP|
    order   = np.argsort(np.abs(sv_for_class))[::-1][:top_n]
    top     = [(pipeline.feature_names[i], float(sv_for_class[i])) for i in order]

    # Build narrative
    narrative_parts = [f"Predicted: {pred_cls.upper()} (passed Stage-1 binary filter)."]
    for feat, sv in top:
        direction = "↑ pushed toward attack" if sv > 0 else "↓ reduced attack score"
        narrative_parts.append(f"  • {feat}: SHAP {sv:+.4f}  ({direction})")
    narrative_parts.append(
        "\nInterpretation: features with the highest |SHAP| value drove this prediction the most."
    )

    return {
        "predicted_class":  pred_cls,
        "stage1_decision":  "attack",
        "top_features":     top,
        "soc_narrative":    "\n".join(narrative_parts),
    }


# ──────────────────────────────────────────────────────────────
# Full evaluation (metrics + latency)
# ──────────────────────────────────────────────────────────────

def evaluate(pipeline: G5Pipeline,
             Xte: np.ndarray,
             yte: np.ndarray,
             feature_names: List[str] | None = None) -> dict:
    """
    Compute all metrics required for the team comparison table.

    Returns a dict with:
      accuracy, macro_f1, per_class_recall (dict),
      normal_false_alarm_rate, predict_ms, explain_ms
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, recall_score, confusion_matrix
    )

    # Predict full test set
    t0    = time.perf_counter()
    preds = predict_pipeline(pipeline, Xte)
    total_ms   = (time.perf_counter() - t0) * 1000
    avg_pred_ms = total_ms / len(Xte)

    # Latency: single-flow predict (100 reps)
    REPS = 100
    t0 = time.perf_counter()
    for _ in range(REPS):
        predict_pipeline(pipeline, Xte[:1])
    single_pred_ms = (time.perf_counter() - t0) * 1000 / REPS

    # Latency: single-flow explain (10 reps — SHAP is slower)
    REPS_EX = 10
    t0 = time.perf_counter()
    for _ in range(REPS_EX):
        explain_flow(pipeline, Xte[0])
    single_explain_ms = (time.perf_counter() - t0) * 1000 / REPS_EX

    all_classes = ["bruteforce", "normal", "scan", "synflood"]

    per_class_recall = {
        c: float(recall_score(yte == c, preds == c, zero_division=0))
        for c in all_classes
    }

    normal_mask = yte == "normal"
    far = float((preds[normal_mask] != "normal").mean()) if normal_mask.sum() > 0 else 0.0

    return {
        "accuracy":               float(accuracy_score(yte, preds)),
        "macro_f1":               float(f1_score(yte, preds, average="macro")),
        "per_class_recall":       per_class_recall,
        "normal_false_alarm_rate": far,
        "single_predict_ms":      round(single_pred_ms, 3),
        "batch_predict_ms_avg":   round(avg_pred_ms, 4),
        "single_explain_ms":      round(single_explain_ms, 2),
        "confusion_matrix":       confusion_matrix(yte, preds, labels=all_classes).tolist(),
        "classes":                all_classes,
    }
