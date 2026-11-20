#!/usr/bin/env python3
"""
G5 Explainable AI — Evaluation on all_flows_v3.csv (Unseen Attack Settings)
=============================================================================
Train on split=="train" (no duplicates).
Test on split=="test_unseen".
Features: all NFStream stats + window features (57 cols). No IPs/ports/timestamps/round/setting.
Methodology: 2-Stage XGBoost (binary→multiclass) + TreeSHAP (Mahmoud 2025, Mohale 2025).
"""

import sys, os, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.expanduser("~"),
    ".gemini/antigravity/brain/cd8b0baf-5c04-47dc-84ca-2a298dfbf152/"
    ".user_uploaded/media_1795142520201_bff242a9.csv",
)
OUT_DIR = Path(__file__).resolve().parent / "results" / "g5_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load & split ───────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH}")

EXCLUDE_COLS = {
    "round", "src_ip", "src_port", "dst_ip", "dst_port",
    "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "label_original", "label", "split", "setting",
}
FEAT_COLS = [c for c in df.columns if c not in EXCLUDE_COLS]
print(f"Feature columns ({len(FEAT_COLS)}): {FEAT_COLS[:5]} ...")

train_df = df[df["split"] == "train"].copy()
test_df  = df[df["split"] == "test_unseen"].copy()

# Drop duplicates if the column exists, otherwise just use all train rows
if "is_duplicate" in df.columns:
    train_df = train_df[train_df["is_duplicate"] == False]
    print("Filtered out duplicate rows using is_duplicate column")
else:
    n_before = len(train_df)
    train_df = train_df.drop_duplicates(subset=FEAT_COLS + ["label"], keep="first")
    n_after = len(train_df)
    print(f"Deduped train: {n_before} → {n_after} rows")

print(f"Train: {len(train_df)}  |  Test-unseen: {len(test_df)}")
print(f"Train labels:\n{train_df['label'].value_counts().to_string()}")
print(f"Test labels:\n{test_df['label'].value_counts().to_string()}")

X_train = train_df[FEAT_COLS].values.astype(np.float32)
y_train = train_df["label"].values
X_test  = test_df[FEAT_COLS].values.astype(np.float32)
y_test  = test_df["label"].values

# ── import G5 pipeline ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from g5_xai_pipeline import train_pipeline, predict_pipeline, explain_flow, evaluate

# ── train ──────────────────────────────────────────────────────────────────
print("\n▶ Training G5 2-stage pipeline …")
t0 = time.perf_counter()
pipe = train_pipeline(X_train, y_train, feature_names=FEAT_COLS)
train_time = time.perf_counter() - t0
print(f"  Training time: {train_time:.2f}s")

# ── predict ────────────────────────────────────────────────────────────────
print("\n▶ Predicting on test_unseen …")
t0 = time.perf_counter()
y_pred = predict_pipeline(pipe, X_test)
pred_time = (time.perf_counter() - t0) / len(X_test) * 1000  # ms per sample
print(f"  Predict latency: {pred_time:.3f} ms/sample")

# ── evaluate ───────────────────────────────────────────────────────────────
print("\n▶ Computing metrics …")
metrics = evaluate(pipe, X_test, y_test, feature_names=FEAT_COLS)
print(json.dumps(metrics, indent=2))

y_pred = predict_pipeline(pipe, X_test)

# ── confusion matrix ──────────────────────────────────────────────────────
from sklearn.metrics import confusion_matrix, classification_report
classes = sorted(set(y_test) | set(y_pred))
cm = confusion_matrix(y_test, y_pred, labels=classes)
print("\n▶ Confusion Matrix:")
print(f"  Labels: {classes}")
for row, lbl in zip(cm, classes):
    print(f"  {lbl:>12s}: {row}")

print("\n▶ Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# ── per-class recall & FAR ────────────────────────────────────────────────
from sklearn.metrics import recall_score, f1_score
per_class_recall = {}
for c in classes:
    mask = y_test == c
    per_class_recall[c] = float(np.mean(y_pred[mask] == c)) if mask.sum() > 0 else 0.0

# FAR = normal samples misclassified as attack / total normal samples
normal_mask = y_test == "normal"
far = float(np.sum(y_pred[normal_mask] != "normal")) / max(1, int(normal_mask.sum()))

macro_f1 = f1_score(y_test, y_pred, average="macro")

print(f"\n▶ Macro-F1: {macro_f1:.4f}")
print(f"▶ FAR (False Alarm Rate): {far:.4f} ({far*100:.2f}%)")
print(f"▶ Per-class Recall: {per_class_recall}")

# ── save metrics JSON ─────────────────────────────────────────────────────
result = {
    "dataset": "all_flows_v3.csv",
    "train_rows": len(train_df),
    "test_rows": len(test_df),
    "n_features": len(FEAT_COLS),
    "macro_f1": round(macro_f1, 4),
    "accuracy": round(float(np.mean(y_pred == y_test)), 4),
    "false_alarm_rate": round(far, 4),
    "per_class_recall": {k: round(v, 4) for k, v in per_class_recall.items()},
    "predict_latency_ms": round(pred_time, 3),
    "train_time_s": round(train_time, 2),
    "confusion_matrix": {"labels": classes, "matrix": cm.tolist()},
}
with open(OUT_DIR / "g5_v3_metrics.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved metrics → {OUT_DIR / 'g5_v3_metrics.json'}")

# ── SHAP plots ────────────────────────────────────────────────────────────
print("\n▶ Generating SHAP plots …")
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Use attack samples for SHAP (stage-2 operates on attacks only)
attack_mask = y_test != "normal"
X_attacks = X_test[attack_mask]
if len(X_attacks) > 200:
    idx = np.random.RandomState(42).choice(len(X_attacks), 200, replace=False)
    X_attacks = X_attacks[idx]

shap_values = pipe.explainer.shap_values(X_attacks)
feat_arr = np.array(FEAT_COLS)

# Bar plot (global importance)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_attacks, feature_names=feat_arr,
                  plot_type="bar", show=False, max_display=15)
plt.title("G5 v3 — SHAP Feature Importance (Attack Classification)")
plt.tight_layout()
plt.savefig(OUT_DIR / "g5_v3_shap_bar.png", dpi=150)
plt.close()
print(f"  Saved → {OUT_DIR / 'g5_v3_shap_bar.png'}")

# Beeswarm plot
plt.figure(figsize=(10, 6))
if isinstance(shap_values, list):
    shap.summary_plot(shap_values[0], X_attacks, feature_names=feat_arr,
                      show=False, max_display=15)
else:
    shap.summary_plot(shap_values, X_attacks, feature_names=feat_arr,
                      show=False, max_display=15)
plt.title("G5 v3 — SHAP Beeswarm (Class 0 Detail)")
plt.tight_layout()
plt.savefig(OUT_DIR / "g5_v3_shap_beeswarm.png", dpi=150)
plt.close()
print(f"  Saved → {OUT_DIR / 'g5_v3_shap_beeswarm.png'}")

# ── Confusion matrix heatmap ─────────────────────────────────────────────
plt.figure(figsize=(7, 5))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("G5 v3 — Confusion Matrix (test_unseen)")
plt.colorbar()
tick_marks = range(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
for i in range(len(classes)):
    for j in range(len(classes)):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(OUT_DIR / "g5_v3_confusion_matrix.png", dpi=150)
plt.close()
print(f"  Saved → {OUT_DIR / 'g5_v3_confusion_matrix.png'}")

# ── Comparison: v3 2-stage vs v3 XGBoost baseline ────────────────────────
print("\n▶ Training XGBoost flat baseline for comparison …")
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

le_map = {c: i for i, c in enumerate(sorted(set(y_train)))}
y_tr_enc = np.array([le_map[l] for l in y_train])
y_te_enc = np.array([le_map[l] for l in y_test])

sw = compute_sample_weight("balanced", y_tr_enc)
baseline = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    tree_method="hist", random_state=42, eval_metric="mlogloss",
    use_label_encoder=False
)
baseline.fit(X_train, y_tr_enc, sample_weight=sw)

inv_map = {v: k for k, v in le_map.items()}
y_base_pred = np.array([inv_map[p] for p in baseline.predict(X_test)])

base_macro_f1 = f1_score(y_test, y_base_pred, average="macro")
base_acc = float(np.mean(y_base_pred == y_test))
base_normal_mask = y_test == "normal"
base_far = float(np.sum(y_base_pred[base_normal_mask] != "normal")) / max(1, int(base_normal_mask.sum()))

base_recall = {}
for c in classes:
    mask = y_test == c
    base_recall[c] = float(np.mean(y_base_pred[mask] == c)) if mask.sum() > 0 else 0.0

print(f"  Baseline Macro-F1: {base_macro_f1:.4f}")
print(f"  Baseline FAR: {base_far:.4f}")
print(f"  Baseline Recall: {base_recall}")

# Comparison table
comp = pd.DataFrame({
    "Metric": ["Macro-F1", "Accuracy", "FAR (%)", 
               "Recall-normal", "Recall-bruteforce", "Recall-scan", "Recall-synflood",
               "Explainability"],
    "G5 2-Stage+SHAP": [
        f"{macro_f1:.4f}", f"{result['accuracy']:.4f}", f"{far*100:.2f}",
        f"{per_class_recall.get('normal',0):.4f}",
        f"{per_class_recall.get('bruteforce',0):.4f}",
        f"{per_class_recall.get('scan',0):.4f}",
        f"{per_class_recall.get('synflood',0):.4f}",
        "TreeSHAP global+local"
    ],
    "XGBoost Baseline": [
        f"{base_macro_f1:.4f}", f"{base_acc:.4f}", f"{base_far*100:.2f}",
        f"{base_recall.get('normal',0):.4f}",
        f"{base_recall.get('bruteforce',0):.4f}",
        f"{base_recall.get('scan',0):.4f}",
        f"{base_recall.get('synflood',0):.4f}",
        "None (black-box)"
    ],
})
comp.to_csv(OUT_DIR / "g5_v3_comparison_table.csv", index=False)
print(f"\n{comp.to_string(index=False)}")
print(f"\nSaved → {OUT_DIR / 'g5_v3_comparison_table.csv'}")

# Comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: main metrics
metrics_names = ["Macro-F1", "Accuracy"]
g5_vals = [macro_f1, result["accuracy"]]
bl_vals = [base_macro_f1, base_acc]
x = np.arange(len(metrics_names))
w = 0.35
axes[0].bar(x - w/2, g5_vals, w, label="G5 2-Stage+SHAP", color="#2196F3")
axes[0].bar(x + w/2, bl_vals, w, label="XGBoost Baseline", color="#FF9800")
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_names)
axes[0].set_ylim(0, 1.05)
axes[0].set_title("Overall Metrics Comparison")
axes[0].legend()
for i, (v1, v2) in enumerate(zip(g5_vals, bl_vals)):
    axes[0].text(i - w/2, v1 + 0.02, f"{v1:.3f}", ha="center", fontsize=9)
    axes[0].text(i + w/2, v2 + 0.02, f"{v2:.3f}", ha="center", fontsize=9)

# Right: per-class recall
attack_classes = [c for c in classes if c != "normal"]
g5_rec = [per_class_recall[c] for c in attack_classes]
bl_rec = [base_recall[c] for c in attack_classes]
x2 = np.arange(len(attack_classes))
axes[1].bar(x2 - w/2, g5_rec, w, label="G5 2-Stage+SHAP", color="#2196F3")
axes[1].bar(x2 + w/2, bl_rec, w, label="XGBoost Baseline", color="#FF9800")
axes[1].set_xticks(x2)
axes[1].set_xticklabels(attack_classes)
axes[1].set_ylim(0, 1.05)
axes[1].set_title("Per-Attack Recall Comparison")
axes[1].legend()
for i, (v1, v2) in enumerate(zip(g5_rec, bl_rec)):
    axes[1].text(i - w/2, v1 + 0.02, f"{v1:.3f}", ha="center", fontsize=9)
    axes[1].text(i + w/2, v2 + 0.02, f"{v2:.3f}", ha="center", fontsize=9)

plt.suptitle("G5 v3 — 2-Stage XGBoost+SHAP vs Flat XGBoost Baseline (test_unseen)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "g5_v3_comparison_chart.png", dpi=150)
plt.close()
print(f"Saved → {OUT_DIR / 'g5_v3_comparison_chart.png'}")

# ── SOC analyst narrative examples ────────────────────────────────────────
print("\n▶ Generating SOC analyst narrative examples …")
attack_indices = np.where(y_test != "normal")[0]
np.random.seed(42)
sample_idx = np.random.choice(attack_indices, min(3, len(attack_indices)), replace=False)

soc_lines = []
for idx in sample_idx:
    xi = X_test[idx: idx+1]
    pred = predict_pipeline(pipe, xi)[0]
    exp = explain_flow(pipe, xi[0])
    soc_lines.append(f"--- Flow #{idx} ---")
    soc_lines.append(f"True Label : {y_test[idx]}")
    soc_lines.append(f"Predicted  : {pred}")
    soc_lines.append(f"Stage-1    : {exp.get('stage1_decision', 'N/A')}")
    soc_lines.append(f"Class      : {exp.get('predicted_class', 'N/A')}")
    soc_lines.append(f"Top Features:")
    for feat_tuple in exp.get("top_features", [])[:5]:
        fname, shap_val = feat_tuple
        soc_lines.append(f"  • {fname}: SHAP={shap_val:.4f}")
    soc_lines.append(f"Narrative  : {exp.get('soc_narrative', 'N/A')}")
    soc_lines.append("")

soc_text = "\n".join(soc_lines)
with open(OUT_DIR / "g5_v3_soc_examples.txt", "w") as f:
    f.write(soc_text)
print(soc_text)
print(f"\nSaved → {OUT_DIR / 'g5_v3_soc_examples.txt'}")

print("\n" + "="*60)
print("G5 v3 EVALUATION COMPLETE")
print("="*60)
