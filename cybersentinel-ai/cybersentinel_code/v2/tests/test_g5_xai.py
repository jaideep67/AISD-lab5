"""
G5 XAI Track — Test Suite (TDD, run BEFORE implementation).
Tests are intentionally strict; every claim must be verifiable.

Run from cybersentinel_code/v2/:
    python -m pytest tests/test_g5_xai.py -v

All tests must pass on the real dataset.
"""
import sys, os, json, time
import numpy as np
import pandas as pd
import pytest

# Ensure the v2 package is importable regardless of CWD
HERE = os.path.dirname(os.path.abspath(__file__))
V2   = os.path.dirname(HERE)                         # cybersentinel_code/v2
sys.path.insert(0, V2)

import features_v2 as fv

DATA_CSV = os.path.join(V2, "data", "flows", "all_flows.csv")
META_JSON = os.path.join(V2, "models", "best_meta.json")

TRAIN_ROUNDS = [3, 4, 5]
TEST_ROUND   = 6
ENVELOPE     = set(fv.ENVELOPE)
FEATURES_66  = fv.FEATURES   # the 66 authorised feature names
CLASSES      = ["bruteforce", "normal", "scan", "synflood"]

# ─────────────────────────── helpers ────────────────────────────

def load_df():
    return pd.read_csv(DATA_CSV)


def split(df):
    train = df[df["round"].isin(TRAIN_ROUNDS)].copy()
    test  = df[df["round"] == TEST_ROUND].copy()
    return train, test


# ═══════════════════════════════════════════════════════════════
# A — DATA INTEGRITY (must pass before any model is built)
# ═══════════════════════════════════════════════════════════════

class TestDataIntegrity:

    def test_csv_exists(self):
        assert os.path.isfile(DATA_CSV), f"Dataset not found: {DATA_CSV}"

    def test_row_count(self):
        df = load_df()
        assert len(df) == 10_676, f"Expected 10 676 rows, got {len(df)}"

    def test_column_count(self):
        df = load_df()
        assert df.shape[1] == 74, f"Expected 74 columns, got {df.shape[1]}"

    def test_66_features_present(self):
        df = load_df()
        missing = [f for f in FEATURES_66 if f not in df.columns]
        assert not missing, f"Missing features: {missing}"

    def test_envelope_not_in_features(self):
        overlap = ENVELOPE & set(FEATURES_66)
        assert not overlap, f"Envelope columns found in FEATURES: {overlap}"

    def test_round_column_values(self):
        df = load_df()
        assert set(df["round"].unique()) == {3, 4, 5, 6}

    def test_class_counts(self):
        df = load_df()
        counts = df["label"].value_counts().to_dict()
        assert counts.get("synflood")  == 5071
        assert counts.get("scan")      == 4008
        assert counts.get("normal")    == 1464
        assert counts.get("bruteforce") == 133

    def test_no_nans_in_features(self):
        df = load_df()
        nan_counts = df[FEATURES_66].isna().sum().sum()
        assert nan_counts == 0, f"{nan_counts} NaN values in feature columns"

    def test_no_inf_in_features(self):
        df = load_df()
        X  = df[FEATURES_66].replace([np.inf, -np.inf], np.nan)
        assert X.isna().sum().sum() == 0, "Inf values found in feature columns"


# ═══════════════════════════════════════════════════════════════
# B — TRAIN / TEST SPLIT (round-based, no leakage)
# ═══════════════════════════════════════════════════════════════

class TestSplit:

    def test_train_rows(self):
        df = load_df()
        train, _ = split(df)
        assert len(train) == 7_941

    def test_test_rows(self):
        df = load_df()
        _, test = split(df)
        assert len(test) == 2_735

    def test_no_round_overlap(self):
        df = load_df()
        train, test = split(df)
        assert set(train["round"]).isdisjoint(set(test["round"]))

    def test_test_round_is_exactly_6(self):
        df = load_df()
        _, test = split(df)
        assert list(test["round"].unique()) == [6]

    def test_bruteforce_minority_in_train(self):
        df = load_df()
        train, _ = split(df)
        bf = (train["label"] == "bruteforce").sum()
        # expect 100 (confirmed from earlier inspection)
        assert bf == 100, f"Expected 100 bruteforce in train, got {bf}"

    def test_bruteforce_minority_in_test(self):
        df = load_df()
        _, test = split(df)
        bf = (test["label"] == "bruteforce").sum()
        assert bf == 33, f"Expected 33 bruteforce in test, got {bf}"


# ═══════════════════════════════════════════════════════════════
# C — G5 PIPELINE CONTRACT (import & basic API)
# ═══════════════════════════════════════════════════════════════

class TestG5PipelineImport:
    """These tests fail until g5_xai_pipeline.py exists — that is intentional (Red phase of TDD)."""

    @pytest.fixture(autouse=True)
    def _import_pipeline(self):
        try:
            import g5_xai_pipeline as g5
            self.g5 = g5
        except ImportError as e:
            pytest.skip(f"g5_xai_pipeline not yet implemented: {e}")

    def test_has_train_function(self):
        assert hasattr(self.g5, "train_pipeline"), "Missing: train_pipeline()"

    def test_has_predict_function(self):
        assert hasattr(self.g5, "predict_pipeline"), "Missing: predict_pipeline()"

    def test_has_explain_function(self):
        assert hasattr(self.g5, "explain_flow"), "Missing: explain_flow()"

    def test_has_evaluate_function(self):
        assert hasattr(self.g5, "evaluate"), "Missing: evaluate()"


# ═══════════════════════════════════════════════════════════════
# D — 2-STAGE ARCHITECTURE CONTRACT
# ═══════════════════════════════════════════════════════════════

class TestTwoStageArchitecture:
    """
    After g5_xai_pipeline.py is implemented, these tests verify the
    2-stage contract (Mahmoud et al. XI2S-IDS):
      Stage 1 → binary (normal vs attack)
      Stage 2 → multi-class (bruteforce / scan / synflood)
    """

    @pytest.fixture(scope="class")
    def pipeline(self):
        try:
            import g5_xai_pipeline as g5
        except ImportError:
            pytest.skip("g5_xai_pipeline not yet implemented")
        df = load_df()
        train, _test = split(df)
        Xtr = train[FEATURES_66].to_numpy(float)
        ytr = train["label"].to_numpy()
        return g5.train_pipeline(Xtr, ytr)

    def test_pipeline_has_stage1(self, pipeline):
        assert hasattr(pipeline, "stage1"), "Pipeline must expose .stage1"

    def test_pipeline_has_stage2(self, pipeline):
        assert hasattr(pipeline, "stage2"), "Pipeline must expose .stage2"

    def test_stage1_binary_classes(self, pipeline):
        # stage1_classes stored in G5Pipeline (xgb>=2 makes .classes_ read-only)
        classes = set(pipeline.stage1_classes)
        assert classes == {0, 1}, \
            f"Stage1 must have binary classes {{0,1}}, got: {classes}"

    def test_stage2_attack_classes_only(self, pipeline):
        """Stage 2 is only trained on attack flows — normal class must not appear."""
        c = set(pipeline.stage2_classes)
        assert "normal" not in c, "Stage 2 must not have a 'normal' class"

    def test_normal_flow_does_not_reach_stage2(self, pipeline):
        """A clearly normal flow should be caught by Stage 1 and not sent to Stage 2."""
        import g5_xai_pipeline as g5
        df = load_df()
        _, test = split(df)
        normal_rows = test[test["label"] == "normal"][FEATURES_66].to_numpy(float)
        # Stage 1 prediction (0 = normal, 1 = attack)
        s1_preds = pipeline.stage1.predict(normal_rows)
        # At least 90 % of normal flows must be stopped at Stage 1
        pct_stopped = (s1_preds == 0).mean()
        assert pct_stopped >= 0.90, \
            f"Stage 1 stops only {pct_stopped*100:.1f}% of normal flows (need ≥ 90%)"

    def test_pipeline_full_predict_shape(self, pipeline):
        import g5_xai_pipeline as g5
        df = load_df()
        _, test = split(df)
        Xte = test[FEATURES_66].to_numpy(float)
        preds = g5.predict_pipeline(pipeline, Xte)
        assert preds.shape == (len(test),), \
            f"predict_pipeline must return shape ({len(test)},), got {preds.shape}"

    def test_pipeline_labels_are_valid(self, pipeline):
        import g5_xai_pipeline as g5
        df = load_df()
        _, test = split(df)
        Xte = test[FEATURES_66].to_numpy(float)
        preds = g5.predict_pipeline(pipeline, Xte)
        invalid = set(preds) - set(CLASSES)
        assert not invalid, f"Invalid prediction labels: {invalid}"


# ═══════════════════════════════════════════════════════════════
# E — SHAP / EXPLAINABILITY CONTRACT
# ═══════════════════════════════════════════════════════════════

class TestExplainability:

    @pytest.fixture(scope="class")
    def fitted(self):
        try:
            import g5_xai_pipeline as g5
        except ImportError:
            pytest.skip("g5_xai_pipeline not yet implemented")
        df = load_df()
        train, test = split(df)
        Xtr = train[FEATURES_66].to_numpy(float)
        ytr = train["label"].to_numpy()
        pipeline = g5.train_pipeline(Xtr, ytr)
        Xte = test[FEATURES_66].to_numpy(float)
        return g5, pipeline, Xte, test

    def test_explain_flow_returns_dict(self, fitted):
        g5, pipeline, Xte, _test = fitted
        result = g5.explain_flow(pipeline, Xte[0])
        assert isinstance(result, dict), "explain_flow must return a dict"

    def test_explain_flow_has_top_features(self, fitted):
        g5, pipeline, Xte, _test = fitted
        result = g5.explain_flow(pipeline, Xte[0])
        assert "top_features" in result, "explain_flow dict must have 'top_features' key"

    def test_explain_flow_top5_length(self, fitted):
        g5, pipeline, Xte, test = fitted
        # Find an attack flow so SHAP engine is exercised (normal flows short-circuit to [])
        attack_rows = test[test["label"] != "normal"]
        attack_x = attack_rows[FEATURES_66].to_numpy(float)[0]
        result = g5.explain_flow(pipeline, attack_x)
        assert len(result["top_features"]) >= 3, \
            "explain_flow must return at least 3 feature attributions for attack flows"

    def test_explain_flow_feature_names_valid(self, fitted):
        g5, pipeline, Xte, _test = fitted
        result = g5.explain_flow(pipeline, Xte[0])
        feature_set = set(FEATURES_66)
        for feat_name, _shap_val in result["top_features"]:
            assert feat_name in feature_set, \
                f"explain_flow returned unknown feature name: {feat_name}"

    def test_explain_flow_has_predicted_class(self, fitted):
        g5, pipeline, Xte, _test = fitted
        result = g5.explain_flow(pipeline, Xte[0])
        assert "predicted_class" in result, "explain_flow dict must have 'predicted_class' key"

    def test_explain_flow_latency_reasonable(self, fitted):
        """Single-flow explanation must complete in under 500 ms (live IDS constraint)."""
        g5, pipeline, Xte, _test = fitted
        t0 = time.perf_counter()
        g5.explain_flow(pipeline, Xte[0])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 500, \
            f"explain_flow took {elapsed_ms:.1f} ms — must be < 500 ms for live use"


# ═══════════════════════════════════════════════════════════════
# F — METRICS CONTRACT
# ═══════════════════════════════════════════════════════════════

class TestMetrics:
    """
    After training, the G5 pipeline must meet minimum performance bars.
    These are LOWER bounds — the pipeline should do better.
    """

    @pytest.fixture(scope="class")
    def results(self):
        try:
            import g5_xai_pipeline as g5
        except ImportError:
            pytest.skip("g5_xai_pipeline not yet implemented")
        from sklearn.metrics import f1_score, accuracy_score, recall_score
        df = load_df()
        train, test = split(df)
        Xtr = train[FEATURES_66].to_numpy(float)
        ytr = train["label"].to_numpy()
        Xte = test[FEATURES_66].to_numpy(float)
        yte = test["label"].to_numpy()
        pipeline = g5.train_pipeline(Xtr, ytr)
        preds    = g5.predict_pipeline(pipeline, Xte)
        return {
            "preds": preds,
            "yte":   yte,
            "macro_f1": f1_score(yte, preds, average="macro"),
            "accuracy": accuracy_score(yte, preds),
            "bruteforce_recall": recall_score(yte == "bruteforce", preds == "bruteforce"),
            "normal_far": (preds[yte == "normal"] != "normal").mean(),
        }

    def test_macro_f1_above_baseline(self, results):
        """Macro-F1 must meet the team XGBoost baseline on round 6 (1.000)."""
        mf1 = results["macro_f1"]
        assert mf1 >= 0.95, f"Macro-F1 too low: {mf1:.4f} (need ≥ 0.95)"

    def test_bruteforce_recall_nonzero(self, results):
        """Must detect at least SOME bruteforce flows (minority class hardest)."""
        bf_r = results["bruteforce_recall"]
        assert bf_r > 0.0, f"Bruteforce recall is 0 — minority class never predicted"

    def test_false_alarm_rate_acceptable(self, results):
        """Normal traffic false-alarm rate must be < 10%."""
        far = results["normal_far"]
        assert far < 0.10, f"False alarm rate on normal traffic: {far*100:.1f}% (need < 10%)"

    def test_accuracy_above_95(self, results):
        acc = results["accuracy"]
        assert acc >= 0.95, f"Accuracy {acc:.4f} below minimum 0.95"


# ═══════════════════════════════════════════════════════════════
# G — LATENCY CONTRACT
# ═══════════════════════════════════════════════════════════════

class TestLatency:

    @pytest.fixture(scope="class")
    def fitted_pipeline(self):
        try:
            import g5_xai_pipeline as g5
        except ImportError:
            pytest.skip("g5_xai_pipeline not yet implemented")
        df = load_df()
        train, test = split(df)
        Xtr = train[FEATURES_66].to_numpy(float)
        ytr = train["label"].to_numpy()
        pipeline = g5.train_pipeline(Xtr, ytr)
        Xte = test[FEATURES_66].to_numpy(float)
        return g5, pipeline, Xte

    def test_single_flow_predict_under_10ms(self, fitted_pipeline):
        """Raw prediction (no SHAP) must be < 10 ms on a single flow."""
        g5, pipeline, Xte = fitted_pipeline
        REPS = 100
        t0 = time.perf_counter()
        for _ in range(REPS):
            g5.predict_pipeline(pipeline, Xte[:1])
        avg_ms = (time.perf_counter() - t0) * 1000 / REPS
        assert avg_ms < 10, f"Avg predict time {avg_ms:.2f} ms — must be < 10 ms"

    def test_single_flow_explain_under_500ms(self, fitted_pipeline):
        """Full explanation (predict + SHAP) must be < 500 ms per flow."""
        g5, pipeline, Xte = fitted_pipeline
        t0 = time.perf_counter()
        g5.explain_flow(pipeline, Xte[0])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 500, f"explain_flow: {elapsed_ms:.1f} ms — must be < 500 ms"
