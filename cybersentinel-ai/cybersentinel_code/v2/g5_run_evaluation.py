"""
G5 XAI — Full Evaluation Runner
Team 10 · Jaideep · 18 Nov 2026

Trains the 2-stage pipeline on rounds 3,4,5; evaluates on round 6.
Produces:
  results/g5/g5_metrics.json             — all metrics
  results/g5/g5_comparison_table.csv     — head-to-head vs baseline
  results/g5/g5_shap_beeswarm.png        — global SHAP summary
  results/g5/g5_shap_bar.png             — mean |SHAP| feature importance
  results/g5/g5_soc_examples.txt         — 3 example SOC explanations
  results/g5/g5_confusion_matrix.png     — confusion matrix

Run from cybersentinel_code/v2/:
    python g5_run_evaluation.py
"""

import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

import features_v2 as fv
from g5_xai_pipeline import train_pipeline, predict_pipeline, explain_flow, evaluate

from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    confusion_matrix, classification_report
)

# ── Paths ──────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(HERE, "data", "flows", "all_flows.csv")
OUT     = os.path.join(HERE, "results", "g5")
os.makedirs(OUT, exist_ok=True)

TRAIN_ROUNDS = [3, 4, 5]
TEST_ROUND   = 6
FEATURES     = fv.FEATURES       # 66 features
CLASSES      = ["bruteforce", "normal", "scan", "synflood"]

# ── Load & Split ───────────────────────────────────────────────
print("=" * 60)
print("G5 — Explainable AI Pipeline   CyberSentinel AI Team 10")
print("=" * 60)
df    = pd.read_csv(DATA)
train = df[df["round"].isin(TRAIN_ROUNDS)]
test  = df[df["round"] == TEST_ROUND]

Xtr = train[FEATURES].to_numpy(float)
ytr = train["label"].to_numpy()
Xte = test[FEATURES].to_numpy(float)
yte = test["label"].to_numpy()

print(f"\nDataset   : {len(df):,} flows  |  {df.shape[1]} columns  |  66 features used")
print(f"Train     : {len(train):,} flows (rounds {TRAIN_ROUNDS})")
print(f"Test      : {len(test):,} flows  (round {TEST_ROUND})")
print(f"\nClass distribution in test:")
for c in CLASSES:
    n = (yte == c).sum()
    print(f"  {c:12s}: {n:4d} ({n/len(yte)*100:.1f}%)")

# ── Train ──────────────────────────────────────────────────────
print("\n[1/5] Training 2-stage pipeline …")
t0_train = time.perf_counter()
pipeline = train_pipeline(Xtr, ytr, feature_names=FEATURES)
train_sec = time.perf_counter() - t0_train
print(f"      Stage 1 (binary XGBoost)   trained in {train_sec:.2f}s")
print(f"      Stage 2 (multiclass XGBoost) classes: {pipeline.stage2_classes}")

# ── Evaluate ───────────────────────────────────────────────────
print("\n[2/5] Evaluating on held-out round 6 …")
metrics = evaluate(pipeline, Xte, yte, FEATURES)

preds = predict_pipeline(pipeline, Xte)

print(f"\n  Accuracy            : {metrics['accuracy']:.4f}")
print(f"  Macro-F1            : {metrics['macro_f1']:.4f}")
print(f"  Normal FAR          : {metrics['normal_false_alarm_rate']*100:.2f}%")
print(f"  Single predict (ms) : {metrics['single_predict_ms']:.3f}")
print(f"  Explain per flow(ms): {metrics['single_explain_ms']:.2f}")
print(f"\n  Per-class recall:")
for c in CLASSES:
    print(f"    {c:12s}: {metrics['per_class_recall'][c]:.4f}")

print("\n  Classification Report:")
print(classification_report(yte, preds, target_names=CLASSES, zero_division=0))

# Save metrics JSON
json.dump(metrics, open(os.path.join(OUT, "g5_metrics.json"), "w"), indent=2)
print(f"  → Saved {os.path.join(OUT, 'g5_metrics.json')}")

# ── Comparison Table ───────────────────────────────────────────
print("\n[3/5] Building comparison table vs baseline …")

# Baseline numbers from PIPELINE_V2.md Phase 4 (measured, not guessed)
baseline = {
    "model":               "XGBoost (baseline, single-stage)",
    "accuracy":            1.0000,
    "macro_f1":            1.0000,
    "bruteforce_recall":   1.000,
    "normal_recall":       1.000,
    "scan_recall":         1.000,
    "synflood_recall":     1.000,
    "normal_false_alarm":  "0.00%",
    "single_predict_ms":   0.67,
    "explain_ms":          "N/A",
    "explainability":      "None",
    "architecture":        "Single-stage multiclass",
}

g5_row = {
    "model":               "G5 XAI — 2-Stage XGBoost + TreeSHAP",
    "accuracy":            round(metrics["accuracy"], 4),
    "macro_f1":            round(metrics["macro_f1"], 4),
    "bruteforce_recall":   round(metrics["per_class_recall"]["bruteforce"], 4),
    "normal_recall":       round(metrics["per_class_recall"]["normal"], 4),
    "scan_recall":         round(metrics["per_class_recall"]["scan"], 4),
    "synflood_recall":     round(metrics["per_class_recall"]["synflood"], 4),
    "normal_false_alarm":  f"{metrics['normal_false_alarm_rate']*100:.2f}%",
    "single_predict_ms":   metrics["single_predict_ms"],
    "explain_ms":          metrics["single_explain_ms"],
    "explainability":      "TreeSHAP (top-5 features per flow)",
    "architecture":        "2-Stage: binary filter → multiclass triage",
}

cmp_df = pd.DataFrame([baseline, g5_row])
cmp_path = os.path.join(OUT, "g5_comparison_table.csv")
cmp_df.to_csv(cmp_path, index=False)
print(f"  → Saved {cmp_path}")
print(cmp_df[["model","macro_f1","bruteforce_recall","normal_false_alarm",
              "single_predict_ms","explain_ms","architecture"]].to_string(index=False))

# ── SHAP Global Plots ──────────────────────────────────────────
print("\n[4/5] Computing global SHAP values for Stage 2 (attack flows only) …")

# Use attack flows from test set for global SHAP
attack_mask = yte != "normal"
Xte_attack  = Xte[attack_mask]
yte_attack  = yte[attack_mask]
print(f"      Using {Xte_attack.shape[0]} attack flows for SHAP summary")

shap_vals = pipeline.explainer.shap_values(Xte_attack)   # (n, 66, 3)

# Mean |SHAP| across all classes
mean_abs_shap = np.abs(shap_vals).mean(axis=(0, 2))       # (66,)
feat_imp = pd.Series(mean_abs_shap, index=FEATURES).sort_values(ascending=False)

# Bar plot — top 20 features
plt.figure(figsize=(10, 7))
feat_imp.head(20).plot(kind="barh", color="steelblue")
plt.gca().invert_yaxis()
plt.xlabel("Mean |SHAP value| across all attack classes")
plt.title("G5 XAI — Global Feature Importance (TreeSHAP)\nTop 20 features (Stage 2, attack flows in Round 6)")
plt.tight_layout()
bar_path = os.path.join(OUT, "g5_shap_bar.png")
plt.savefig(bar_path, dpi=130); plt.close()
print(f"  → Saved {bar_path}")

# Beeswarm (SHAP summary plot) for the most common attack class (scan)
scan_idx   = pipeline.stage2_classes.index("scan")
shap_scan  = shap_vals[:, :, scan_idx]
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_scan, Xte_attack, feature_names=FEATURES,
    max_display=20, show=False, plot_size=None
)
plt.title("G5 XAI — SHAP Beeswarm (Stage 2: SCAN class)", pad=10)
plt.tight_layout()
bee_path = os.path.join(OUT, "g5_shap_beeswarm.png")
plt.savefig(bee_path, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → Saved {bee_path}")

# ── Confusion Matrix ───────────────────────────────────────────
cm = confusion_matrix(yte, preds, labels=CLASSES)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=35, ha="right")
ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"G5 XAI Confusion Matrix\n(2-Stage Pipeline, Test = Round 6)")
plt.colorbar(im, ax=ax); plt.tight_layout()
cm_path = os.path.join(OUT, "g5_confusion_matrix.png")
plt.savefig(cm_path, dpi=130); plt.close()
print(f"  → Saved {cm_path}")

# ── SOC Analyst Example Explanations ──────────────────────────
print("\n[5/5] Generating SOC analyst narrative examples …")

soc_lines = ["G5 XAI — SOC Analyst Explanation Examples", "=" * 60, ""]
examples = {
    "SCAN": np.where(yte == "scan")[0][0],
    "SYNFLOOD": np.where(yte == "synflood")[0][0],
    "BRUTEFORCE": np.where(yte == "bruteforce")[0][0] if (yte == "bruteforce").any() else None,
    "NORMAL": np.where(yte == "normal")[0][0],
}

for label, idx in examples.items():
    if idx is None:
        continue
    soc_lines.append(f"─── Example: TRUE LABEL = {label} (flow index {idx}) ───")
    result = explain_flow(pipeline, Xte[idx])
    soc_lines.append(f"Stage-1 decision  : {result['stage1_decision'].upper()}")
    soc_lines.append(f"Final prediction  : {result['predicted_class'].upper()}")
    soc_lines.append(f"Correct?          : {'✓ YES' if result['predicted_class'] == yte[idx] else '✗ NO (misclassified)'}")
    if result["top_features"]:
        soc_lines.append("Top SHAP drivers  :")
        for feat, sv in result["top_features"]:
            direction = "↑" if sv > 0 else "↓"
            soc_lines.append(f"  {direction}  {feat:35s}  SHAP = {sv:+.4f}")
    soc_lines.append(f"\nSOC Narrative:\n{result['soc_narrative']}")
    soc_lines.append("")

soc_path = os.path.join(OUT, "g5_soc_examples.txt")
with open(soc_path, "w") as f:
    f.write("\n".join(soc_lines))
print(f"  → Saved {soc_path}")

# ── Final Summary ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("G5 EVALUATION COMPLETE — REAL MEASURED NUMBERS")
print("=" * 60)
print(f"  Macro-F1              : {metrics['macro_f1']:.4f}")
print(f"  Accuracy              : {metrics['accuracy']:.4f}")
print(f"  Bruteforce recall     : {metrics['per_class_recall']['bruteforce']:.4f}  ← hardest minority class")
print(f"  Scan recall           : {metrics['per_class_recall']['scan']:.4f}")
print(f"  Synflood recall       : {metrics['per_class_recall']['synflood']:.4f}")
print(f"  Normal recall         : {metrics['per_class_recall']['normal']:.4f}")
print(f"  Normal FAR            : {metrics['normal_false_alarm_rate']*100:.2f}%")
print(f"  Single predict (ms)   : {metrics['single_predict_ms']:.3f}  ← Stage 1 + 2")
print(f"  Single explain (ms)   : {metrics['single_explain_ms']:.2f}  ← predict + SHAP")
print(f"\n  Output files in       : {OUT}/")
print(f"    g5_metrics.json")
print(f"    g5_comparison_table.csv")
print(f"    g5_shap_beeswarm.png")
print(f"    g5_shap_bar.png")
print(f"    g5_confusion_matrix.png")
print(f"    g5_soc_examples.txt")
print("=" * 60)
