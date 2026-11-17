# G5 — Explainable AI (XAI) & 2-Stage Filter: Results & Comparative Analysis Report
**CyberSentinel AI · Team 10 · Bachu Venkata Siva Jaideep (01fe24bci084)**

---

## Executive Summary

This report presents the implementation and verified empirical evaluation of **Track G5 (Explainable AI & Multi-Stage IDS)** for the CyberSentinel AI live intrusion detection project. Grounded in two recent 2025 research publications—**Mahmoud et al. (XI2S-IDS, *Future Internet* 2025)** and **Mohale & Obagbuwa (*Frontiers in Computer Science* 2025)**—Track G5 addresses the two fundamental limitations of the baseline IDS:
1. **Black-box opacity**: Security analysts need to know *why* an alert was generated before taking remediation actions.
2. **Computational waste**: Monolithic multi-class models evaluate all class boundaries for every benign flow.

Our 2-stage hierarchical pipeline with **TreeSHAP local and global explainability** achieves **$1.0000$ Macro-F1**, maintains **$100\%$ recall on the extreme minority Brute Force class (33/33 flows)** with **$0.00\%$ false alarms**, and runs **$9.05\times$ faster** than the baseline model ($0.074\text{ ms/flow}$ vs $0.670\text{ ms/flow}$), with an on-demand explanation overhead of only **$0.060\text{ ms/flow}$**.

---

## 1. Comparative Analysis: Track G5 vs. XGBoost Baseline

The table below contrasts our G5 2-Stage XAI model directly against the team's single-stage XGBoost baseline on the held-out **Round 6 test dataset (2,735 flows)**:

| Evaluation Dimension | Baseline XGBoost (1-Stage) | Track G5: 2-Stage + TreeSHAP | Advantage / Differentiator |
|---|---|---|---|
| **Architecture** | Monolithic 4-class model | 2-Stage: Binary Filter $\rightarrow$ Multi-Class Triage | Clean separation of detection & triage |
| **Research Grounding** | Standard ML practice | Mahmoud et al. (2025) & Mohale et al. (2025) | Literature-backed methodology |
| **Overall Accuracy** | **1.0000** ($2,735/2,735$) | **1.0000** ($2,735/2,735$) | Identical top performance |
| **Macro-F1 Score** | **1.0000** | **1.0000** | Identical top performance |
| **Bruteforce Recall** | **1.0000** ($33/33$) | **1.0000** ($33/33$) | 100% minority detection preserved |
| **Port Scan Recall** | **1.0000** ($1,002/1,002$) | **1.0000** ($1,002/1,002$) | 100% detection |
| **SYN Flood Recall** | **1.0000** ($1,255/1,255$) | **1.0000** ($1,255/1,255$) | 100% detection |
| **Normal Recall** | **1.0000** ($445/445$) | **1.0000** ($445/445$) | 100% pass-through |
| **Normal False Alarm Rate (FAR)** | **0.00%** ($0/445$) | **0.00%** ($0/445$) | Zero false alarms on benign flows |
| **Single-Flow Prediction Latency** | $0.670\text{ ms}$ | **$0.074\text{ ms}$** | **$9.05\times$ Faster** (Benign traffic short-circuits) |
| **Batch Inference Time (Avg/flow)**| $0.011\text{ ms}$ | **$0.003\text{ ms}$** | **$3.66\times$ Faster** |
| **Explanation Latency** | N/A (Cannot explain) | **$0.060\text{ ms/flow}$** | Instantaneous real-time SOC attribution |
| **Explainability Level** | ❌ Black box (raw string only) | ✅ **TreeSHAP Top-5 feature attribution + SOC narrative** | Actionable for security operations |
| **Features Used** | 22 selected features | Exactly 66 authorized flow features | Uses full statistical & window profile |
| **Metadata Protection** | Excluded | Excluded ($0\%$ leakage of IPs/ports/times) | Fair evaluation compliance |

### Visual Comparison Plot
![Track G5 vs Baseline Comparison](images/g5_vs_baseline_comparison.png)

---

## 2. In-Depth Technical Breakdown of the Speedup & Architecture

### Why is G5 $9\times$ Faster than the Baseline?
In a standard live network deployment, over **$80\text{--}90\%$ of network flows are benign**. In the baseline single-stage XGBoost model, every single flow must evaluate all trees for all 4 classes ($M$ trees $\times K$ classes), requiring multiple vector dot-products and branch checks.

Under the **Mahmoud et al. (XI2S-IDS)** 2-stage hierarchical architecture:
1. **Stage 1 (Binary Filter)** is trained strictly as a shallow, optimized binary classifier (`normal = 0`, `attack = 1`). Normal traffic short-circuits immediately in **$0.074\text{ ms}$** and is marked `OK`. Stage 2 and the SHAP engine are never invoked for normal flows.
2. **Stage 2 (Multi-Class Triage)** is executed **only when an anomaly is confirmed**, routing compute power to classifying whether the attack is `bruteforce`, `scan`, or `synflood`.

```
Incoming Flow (66 Features)
           │
           ▼
┌────────────────────────────────────────┐
│  STAGE 1: Coarse Binary Filter         │  <-- 0.074 ms latency
│  (Mahmoud et al. XI2S-IDS)             │
└──────────────────┬─────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
    [ Normal ]          [ Attack ]
    (Pass through)           │
                             ▼
                ┌────────────────────────────────────────┐
                │  STAGE 2: Multi-Class Attack Classifier│  <-- Triage: Scan vs Flood vs Brute Force
                │  (XGBoost with class-weights)          │
                └──────────────────┬─────────────────────┘
                                   │
                                   ▼
                ┌────────────────────────────────────────┐
                │  XAI EXPLAINABILITY ENGINE (TreeSHAP)  │  <-- 0.060 ms on-demand
                │  1. Top-5 feature attribution per flow │
                │  2. Global feature beeswarm & summary  │
                │  3. Plain-English SOC Analyst triage   │
                └────────────────────────────────────────┘
```

---

## 3. Explainable AI (TreeSHAP) Insights

Traditional explainability methods like LIME or KernelSHAP require perturbation of hundreds of synthetic points, adding hundreds of milliseconds of delay—unusable for high-speed networking. Following **Mohale & Obagbuwa (2025)**, we employ **TreeSHAP**, which computes exact game-theoretic Shapley values by traversing the tree ensemble graph in $O(TLD^2)$, executing in only **$0.060\text{ ms}$**.

### A. Global Feature Importance (Top-20 Attack Discriminators)
The bar chart below ranks the features that contribute most heavily across all attack classifications in Round 6:

![SHAP Feature Importance Bar](images/g5_shap_bar.png)

**Key Observations:**
1. **`bidirectional_mean_ps` (Mean Packet Size):** Dominates classification for volumetric floods and brute-force attempts. SYN floods feature consistently small 40-to-60-byte packets, whereas legitimate browsing and SSH sessions contain variable-length payloads.
2. **`src2dst_bytes` & `dst2src_min_ps`:** Clearly isolate port scans, where scanners transmit minimal payload bytes across thousands of outbound probes while receiving empty RST/ACK teardown packets.
3. **`win_dst_ports_10s` & `win_half_open_10s`:** Capture rapid horizontal scanning and incomplete three-way handshakes over 10-second rolling windows.

### B. SHAP Beeswarm Plot (Port Scan Category)
The beeswarm plot displays feature distribution impacts for individual flows:

![SHAP Beeswarm Plot](images/g5_shap_beeswarm.png)

High values of `src2dst_bytes` and high `src2dst_mean_piat_ms` (packet inter-arrival times) consistently push the model's log-odds output toward the `SCAN` decision boundary.

---

## 4. Confusion Matrix on Held-Out Test Set (Round 6)

The confusion matrix confirms zero off-diagonal misclassifications across all 2,735 flows in Round 6:

![Confusion Matrix](images/g5_confusion_matrix.png)

```
Classification Report:
              precision    recall  f1-score   support

  bruteforce       1.00      1.00      1.00        33
      normal       1.00      1.00      1.00       445
        scan       1.00      1.00      1.00      1002
    synflood       1.00      1.00      1.00      1255

    accuracy                           1.00      2735
   macro avg       1.00      1.00      1.00      2735
weighted avg       1.00      1.00      1.00      2735
```

---

## 5. Live SOC Analyst Explanation Samples

The pipeline produces structured, human-readable narratives for live SOC feeds. Excerpted from `results/g5/g5_soc_examples.txt`:

### Case 1: Port Scan Detection
```text
Stage-1 Decision  : ATTACK
Final Prediction  : SCAN [✓ Correct]
Top Contributing Features:
  ↑  src2dst_bytes          SHAP = +3.4636  (High outbound bytes matching probe sweep)
  ↓  win_dst_ports_10s      SHAP = -1.2043  (Port dispersion suppresses flood category)
  ↑  src2dst_mean_piat_ms   SHAP = +0.2252  (Inter-arrival rate matches nmap scan cadence)
  ↑  bidirectional_packets  SHAP = +0.0870
  ↓  dst2src_min_ps         SHAP = -0.0025

SOC Narrative:
"Predicted: SCAN (Passed Stage-1 binary filter). Primary trigger: src2dst_bytes (+3.46)
combined with high unique destination port touches over 10s window."
```

### Case 2: SYN Flood Detection
```text
Stage-1 Decision  : ATTACK
Final Prediction  : SYNFLOOD [✓ Correct]
Top Contributing Feature:
  ↑  bidirectional_mean_ps  SHAP = +4.8415  (Uniform tiny 40-54 byte SYN packets)

SOC Narrative:
"Predicted: SYNFLOOD. Extremely low mean packet size with absent ACK responses
confirms half-open connection flood."
```

### Case 3: Brute Force Password Guessing (Minority Class — 33 flows)
```text
Stage-1 Decision  : ATTACK
Final Prediction  : BRUTEFORCE [✓ Correct]
Top Contributing Feature:
  ↑  bidirectional_mean_ps  SHAP = +4.8543  (Fixed-size repeated SSH authentication payload)

SOC Narrative:
"Predicted: BRUTEFORCE. Repeated fixed-length packet bursts directed to port 22
exhibits characteristic Hydra SSH dictionary attack profile."
```

### Case 4: Normal Benign Traffic
```text
Stage-1 Decision  : NORMAL
Final Prediction  : NORMAL [✓ Correct]
SOC Narrative:
"Flow classified as NORMAL by Stage-1 filter. Short-circuited with zero triage delay."
```

---

## 6. Verification and Automated Test Results

Running `pytest tests/test_g5_xai.py -v` verifies all **38 unit tests**:

```text
tests/test_g5_xai.py::TestDataIntegrity::test_csv_exists PASSED               [  2%]
tests/test_g5_xai.py::TestDataIntegrity::test_row_count PASSED                [  5%]
tests/test_g5_xai.py::TestDataIntegrity::test_column_count PASSED             [  7%]
tests/test_g5_xai.py::TestDataIntegrity::test_66_features_present PASSED      [ 10%]
tests/test_g5_xai.py::TestDataIntegrity::test_envelope_not_in_features PASSED  [ 13%]
tests/test_g5_xai.py::TestDataIntegrity::test_round_column_values PASSED      [ 15%]
tests/test_g5_xai.py::TestDataIntegrity::test_class_counts PASSED             [ 18%]
tests/test_g5_xai.py::TestDataIntegrity::test_no_nans_in_features PASSED      [ 21%]
tests/test_g5_xai.py::TestDataIntegrity::test_no_inf_in_features PASSED       [ 23%]
tests/test_g5_xai.py::TestSplit::test_train_rows PASSED                       [ 26%]
tests/test_g5_xai.py::TestSplit::test_test_rows PASSED                        [ 28%]
tests/test_g5_xai.py::TestSplit::test_no_round_overlap PASSED                 [ 31%]
tests/test_g5_xai.py::TestSplit::test_test_round_is_exactly_6 PASSED          [ 34%]
tests/test_g5_xai.py::TestSplit::test_bruteforce_minority_in_train PASSED    [ 36%]
tests/test_g5_xai.py::TestSplit::test_bruteforce_minority_in_test PASSED     [ 39%]
tests/test_g5_xai.py::TestG5PipelineImport::test_has_train_function PASSED    [ 42%]
tests/test_g5_xai.py::TestG5PipelineImport::test_has_predict_function PASSED  [ 44%]
tests/test_g5_xai.py::TestG5PipelineImport::test_has_explain_function PASSED  [ 47%]
tests/test_g5_xai.py::TestG5PipelineImport::test_has_evaluate_function PASSED [ 50%]
tests/test_g5_xai.py::TestTwoStageArchitecture::test_pipeline_has_stage1 PASSED [ 52%]
tests/test_g5_xai.py::TestTwoStageArchitecture::test_pipeline_has_stage2 PASSED [ 55%]
tests/test_g5_xai.py::TestTwoStageArchitecture::test_stage1_binary_classes PASSED [ 57%]
tests/test_g5_xai.py::TestTwoStageArchitecture::test_stage2_attack_classes_only PASSED [ 60%]
tests/test_g5_xai.py::TestTwoStageArchitecture::test_normal_flow_does_not_reach_stage2 PASSED [ 63%]
tests/test_g5_xai.py::TestTwoStageArchitecture::test_pipeline_full_predict_shape PASSED [ 65%]
tests/test_g5_xai.py::TestTwoStageArchitecture::test_pipeline_labels_are_valid PASSED [ 68%]
tests/test_g5_xai.py::TestExplainability::test_explain_flow_returns_dict PASSED [ 71%]
tests/test_g5_xai.py::TestExplainability::test_explain_flow_has_top_features PASSED [ 73%]
tests/test_g5_xai.py::TestExplainability::test_explain_flow_top5_length PASSED [ 76%]
tests/test_g5_xai.py::TestExplainability::test_explain_flow_feature_names_valid PASSED [ 78%]
tests/test_g5_xai.py::TestExplainability::test_explain_flow_has_predicted_class PASSED [ 81%]
tests/test_g5_xai.py::TestExplainability::test_explain_flow_latency_reasonable PASSED [ 84%]
tests/test_g5_xai.py::TestMetrics::test_macro_f1_above_baseline PASSED        [ 86%]
tests/test_g5_xai.py::TestMetrics::test_bruteforce_recall_nonzero PASSED      [ 89%]
tests/test_g5_xai.py::TestMetrics::test_false_alarm_rate_acceptable PASSED   [ 92%]
tests/test_g5_xai.py::TestMetrics::test_accuracy_above_95 PASSED              [ 94%]
tests/test_g5_xai.py::TestLatency::test_single_flow_predict_under_10ms PASSED  [ 97%]
tests/test_g5_xai.py::TestLatency::test_single_flow_explain_under_500ms PASSED [100%]

============================== 38 passed in 4.87s ==============================
```

---

## 7. How to Reproduce

All files are self-contained in `cybersentinel-ai/`:

```bash
cd cybersentinel-ai/cybersentinel_code/v2

# 1. Run all 38 tests
python -m pytest tests/test_g5_xai.py -v

# 2. Re-train and regenerate all evaluation metrics & plots
python g5_run_evaluation.py
```
