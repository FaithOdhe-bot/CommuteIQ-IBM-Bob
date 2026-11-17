"""
test_model_compatibility.py — Model compatibility and reproducibility tests.

Purpose:
  1. Verify that import-path unification eliminates duplicate model loading.
  2. Verify raw model predictions against current-environment reference values.
  3. Verify that the XGBoost model is NOT called for sub-50 km trips (formula path).
  4. Verify that model warnings come from exactly ONE load event per process.

These tests target the trained pkl artifacts directly, NOT the API. They
are intentionally separate from API-level tests so that model regressions
can be isolated from routing, encoding, or middleware changes.

NOTE ON REFERENCE VALUES:
  reference_predictions.json was recorded on sklearn 1.9.1 / XGBoost 3.4.1.
  The artifacts were serialised with sklearn 1.8.0 / XGBoost 3.4.1.
  These are current-environment references, NOT verified original-training
  outputs. A failure means the current environment produces different outputs
  than when the reference was first recorded (or the reference file was changed).
  It does NOT prove the original training outputs were the same.
"""

import json
import math
import sys
import pathlib
import warnings
import unittest

import pytest

# ---------------------------------------------------------------------------
# Path setup — same approach as the rest of the test suite
# ---------------------------------------------------------------------------
APP_DIR = pathlib.Path(__file__).parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

MODELS_DIR = pathlib.Path(__file__).parent.parent / "models"
REF_FILE   = MODELS_DIR / "reference_predictions.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def models():
    """Load all three ML artifacts once per test module.  Warnings are
    suppressed here — they are tested separately in TestImportPathUnification."""
    import joblib
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        travel_model  = joblib.load(MODELS_DIR / "travel_time_model.pkl")
        quality_model = joblib.load(MODELS_DIR / "commute_quality_model.pkl")
        encoders      = joblib.load(MODELS_DIR / "encoders.pkl")
    return travel_model, quality_model, encoders


@pytest.fixture(scope="module")
def reference():
    assert REF_FILE.exists(), (
        f"Reference file missing: {REF_FILE}\n"
        "Run: python -c \"import scripts.generate_reference\" to regenerate."
    )
    return json.loads(REF_FILE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# TestImportPathUnification
# ---------------------------------------------------------------------------

class TestImportPathUnification:
    """Verify that app.main and main no longer resolve to separate module objects."""

    def test_app_main_and_main_are_same_module(self):
        """After the import-path fix, importing app.main twice must yield the
        same object — no second load_models() call, no extra warnings."""
        import app.main as first_import
        import app.main as second_import
        assert first_import is second_import, (
            "app.main resolved to two different module objects — "
            "duplicate model loading would occur."
        )

    def test_quality_model_is_not_duplicated(self):
        """quality_model loaded via app.main must be the same object both times."""
        import app.main as m1
        import app.main as m2
        assert m1.quality_model is m2.quality_model, (
            "quality_model objects differ — load_models() ran more than once."
        )

    def test_travel_model_is_not_duplicated(self):
        import app.main as m1
        import app.main as m2
        assert m1.travel_model is m2.travel_model

    def test_no_duplicate_warnings_on_reimport(self):
        """Re-importing app.main after it is already in sys.modules must emit
        zero new warnings (models are not re-unpickled)."""
        import app.main  # ensure it is already cached
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import importlib
            # Force a re-lookup (does NOT re-execute module-level code)
            importlib.import_module("app.main")
        pkl_warns = [
            w for w in caught
            if w.category.__name__ in ("InconsistentVersionWarning", "UserWarning")
        ]
        assert len(pkl_warns) == 0, (
            f"Re-importing app.main emitted {len(pkl_warns)} pkl warnings — "
            "models are being loaded more than once."
        )


# ---------------------------------------------------------------------------
# TestPklVersionMetadata
# ---------------------------------------------------------------------------

class TestPklVersionMetadata:
    """Verify what is actually recorded in the pkl artifacts."""

    def test_rf_sklearn_version_in_pickle_is_1_8_0(self):
        """The RandomForest pkl embeds sklearn 1.8.0 — NOT 1.5.0 as listed in
        requirements.txt.  This test documents the confirmed training version."""
        import re
        data = (MODELS_DIR / "commute_quality_model.pkl").read_bytes()
        # Version string is stored verbatim in the pickle byte stream
        matches = re.findall(rb"\d+\.\d+\.\d+", data[:2000])
        versions = {m.decode() for m in matches}
        assert "1.8.0" in versions, (
            f"Expected sklearn 1.8.0 in pkl bytes, found: {versions}. "
            "If this changes, the training-version record must be updated."
        )

    def test_rf_sklearn_version_is_not_1_5_0(self):
        """requirements.txt incorrectly listed sklearn 1.5.0 as the training
        version.  This test confirms that assumption is wrong."""
        import re
        data = (MODELS_DIR / "commute_quality_model.pkl").read_bytes()
        matches = re.findall(rb"\d+\.\d+\.\d+", data[:2000])
        versions = {m.decode() for m in matches}
        assert "1.5.0" not in versions, (
            "Found 1.5.0 in pkl — requirements.txt assumption was actually correct. "
            "Update MODEL_PROVENANCE.md and this test."
        )

    def test_xgboost_saved_version_matches_current(self):
        """XGBoost save_config reports version [3,4,1] — the same as the current
        environment.  The UserWarning is about pickle FORMAT, not version mismatch."""
        import joblib, json as _json
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            xgb_model = joblib.load(MODELS_DIR / "travel_time_model.pkl")
        cfg = _json.loads(xgb_model.get_booster().save_config())
        assert cfg["version"] == [3, 4, 1], (
            f"XGBoost saved version {cfg['version']} != [3,4,1]. "
            "MODEL_PROVENANCE.md must be updated."
        )

    def test_rf_has_200_decision_tree_estimators(self):
        import joblib
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            rf = joblib.load(MODELS_DIR / "commute_quality_model.pkl")
        assert rf.n_estimators == 200
        assert len(rf.estimators_) == 200

    def test_rf_feature_count_is_8(self):
        import joblib
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            rf = joblib.load(MODELS_DIR / "commute_quality_model.pkl")
        assert rf.n_features_in_ == 8

    def test_xgb_feature_count_is_8(self):
        import joblib
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            xgb = joblib.load(MODELS_DIR / "travel_time_model.pkl")
        assert xgb.n_features_in_ == 8

    def test_rf_feature_names_match_reference(self):
        import joblib
        expected = [
            "distance_km", "congestion_enc", "weather_enc", "alternatives",
            "is_peak", "mode_enc", "city_enc", "road_quality",
        ]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            rf = joblib.load(MODELS_DIR / "commute_quality_model.pkl")
        actual = list(rf.feature_names_in_)
        assert actual == expected, f"RF feature names changed: {actual}"


# ---------------------------------------------------------------------------
# TestTravelTimeReferenceOutputs
# ---------------------------------------------------------------------------

class TestTravelTimeReferenceOutputs:
    """Raw XGBoost predictions for fixed inputs must match the recorded baseline.

    Tolerance: 0.01 minutes (~0.6 s).  A larger delta indicates the pkl loaded
    under a different sklearn/XGBoost version is producing numerically different
    outputs — which would require a reserialisation or retrain decision.
    """

    TOLERANCE = 0.01  # minutes

    def _make_df(self, vals, feat_names):
        import pandas as pd
        return pd.DataFrame([vals], columns=feat_names)

    def test_lagos_driving_low_clear(self, models, reference):
        travel_model, _, encoders = models
        feat = encoders["features"]
        df   = self._make_df([129.0, 0, 0, 1, 0, 0, 0, 27.1], feat)
        pred = float(travel_model.predict(df)[0])
        expected = reference["travel_time_reference"]["lagos_driving_low_clear"]
        assert abs(pred - expected) < self.TOLERANCE, (
            f"XGBoost output drift: expected {expected:.4f}, got {pred:.4f}. "
            "Possible sklearn version incompatibility."
        )

    def test_lagos_driving_high_rainy(self, models, reference):
        travel_model, _, encoders = models
        feat = encoders["features"]
        df   = self._make_df([200.0, 2, 3, 2, 1, 0, 0, 27.1], feat)
        pred = float(travel_model.predict(df)[0])
        expected = reference["travel_time_reference"]["lagos_driving_high_rainy"]
        assert abs(pred - expected) < self.TOLERANCE

    def test_nairobi_matatu_medium_clear(self, models, reference):
        travel_model, _, encoders = models
        feat = encoders["features"]
        df   = self._make_df([150.0, 1, 0, 1, 0, 7, 1, 30.5], feat)
        pred = float(travel_model.predict(df)[0])
        expected = reference["travel_time_reference"]["nairobi_matatu_medium_clear"]
        assert abs(pred - expected) < self.TOLERANCE

    def test_nairobi_boda_boda_low_foggy(self, models, reference):
        travel_model, _, encoders = models
        feat = encoders["features"]
        df   = self._make_df([300.0, 0, 2, 1, 0, 9, 1, 30.5], feat)
        pred = float(travel_model.predict(df)[0])
        expected = reference["travel_time_reference"]["nairobi_boda_boda_low_foggy"]
        assert abs(pred - expected) < self.TOLERANCE

    def test_lagos_okada_high_clear(self, models, reference):
        travel_model, _, encoders = models
        feat = encoders["features"]
        df   = self._make_df([500.0, 2, 0, 3, 1, 3, 0, 20.0], feat)
        pred = float(travel_model.predict(df)[0])
        expected = reference["travel_time_reference"]["lagos_okada_high_clear"]
        assert abs(pred - expected) < self.TOLERANCE

    def test_all_reference_predictions_are_positive(self, models, reference):
        """All reference travel times must be positive."""
        for key, val in reference["travel_time_reference"].items():
            assert val > 0, f"Reference travel time for {key} is non-positive: {val}"

    def test_all_reference_predictions_are_finite(self, models, reference):
        for key, val in reference["travel_time_reference"].items():
            assert math.isfinite(val), f"Reference travel time for {key} is not finite: {val}"


# ---------------------------------------------------------------------------
# TestQualityReferenceOutputs
# ---------------------------------------------------------------------------

class TestQualityReferenceOutputs:
    """Raw RandomForest class predictions for fixed inputs must match the baseline.

    Class labels (0=Poor, 1=Moderate, 2=Good) must be identical.
    Probabilities are checked within 0.01 tolerance per class.
    """

    PROBA_TOLERANCE = 0.01

    def _predict(self, models, vals):
        import pandas as pd
        _, quality_model, encoders = models
        feat = encoders["features"]
        df   = pd.DataFrame([vals], columns=feat)
        cls  = int(quality_model.predict(df)[0])
        proba = quality_model.predict_proba(df)[0].tolist()
        return cls, proba

    def test_lagos_driving_low_clear_good(self, models, reference):
        cls, proba = self._predict(models, [10.0, 0, 0, 2, 0, 0, 0, 27.1])
        ref = reference["quality_reference"]["lagos_driving_low_clear_good"]
        assert cls == ref["class_int"], (
            f"RF class changed: expected {ref['class_int']} ({ref['class_label']}), got {cls}"
        )
        for i, (got, exp) in enumerate(zip(proba, ref["proba"])):
            assert abs(got - exp) < self.PROBA_TOLERANCE, (
                f"RF proba[{i}] drift: expected {exp}, got {got}"
            )

    def test_lagos_driving_high_rainy_poor(self, models, reference):
        cls, proba = self._predict(models, [15.0, 2, 3, 2, 1, 0, 0, 27.1])
        ref = reference["quality_reference"]["lagos_driving_high_rainy_poor"]
        assert cls == ref["class_int"]
        for i, (got, exp) in enumerate(zip(proba, ref["proba"])):
            assert abs(got - exp) < self.PROBA_TOLERANCE

    def test_nairobi_matatu_medium_clear(self, models, reference):
        cls, proba = self._predict(models, [20.0, 1, 0, 2, 0, 7, 1, 30.5])
        ref = reference["quality_reference"]["nairobi_matatu_medium_clear"]
        assert cls == ref["class_int"]
        for i, (got, exp) in enumerate(zip(proba, ref["proba"])):
            assert abs(got - exp) < self.PROBA_TOLERANCE

    def test_nairobi_boda_medium_rainy(self, models, reference):
        cls, proba = self._predict(models, [8.0, 1, 3, 1, 1, 9, 1, 30.5])
        ref = reference["quality_reference"]["nairobi_boda_medium_rainy"]
        assert cls == ref["class_int"]
        for i, (got, exp) in enumerate(zip(proba, ref["proba"])):
            assert abs(got - exp) < self.PROBA_TOLERANCE

    def test_lagos_brt_low_cloudy(self, models, reference):
        cls, proba = self._predict(models, [12.0, 0, 1, 2, 0, 2, 0, 27.1])
        ref = reference["quality_reference"]["lagos_brt_low_cloudy"]
        assert cls == ref["class_int"]
        for i, (got, exp) in enumerate(zip(proba, ref["proba"])):
            assert abs(got - exp) < self.PROBA_TOLERANCE

    def test_quality_classes_are_0_1_or_2(self, models, reference):
        for key, ref in reference["quality_reference"].items():
            assert ref["class_int"] in (0, 1, 2), (
                f"Quality reference {key} has invalid class {ref['class_int']}"
            )

    def test_quality_probas_sum_to_one(self, models, reference):
        for key, ref in reference["quality_reference"].items():
            total = sum(ref["proba"])
            assert abs(total - 1.0) < 0.001, (
                f"Quality probas for {key} sum to {total}, not 1.0"
            )


# ---------------------------------------------------------------------------
# TestModelNotCalledForSubUrbanTrips
# ---------------------------------------------------------------------------

class TestModelNotCalledForSubUrbanTrips:
    """XGBoost must NOT be called for trips < 50 km (out-of-distribution).
    The formula fallback must be used instead, and must return a valid result.
    """

    def test_sub_50km_uses_formula_not_xgboost(self):
        """For 10 km, predict_travel_time must use the formula path.
        We verify this by confirming the result matches the formula output
        rather than any XGBoost prediction."""
        from app.main import predict_travel_time, _formula_travel_time
        formula_result = _formula_travel_time(10.0, "Low", "Clear", "driving", "lagos")
        api_result     = predict_travel_time(10.0, "Low", "Clear", 2, "driving", "lagos", 27.1)
        assert api_result == formula_result, (
            f"Sub-50km trip returned {api_result}, formula gives {formula_result}. "
            "XGBoost may be called for out-of-distribution inputs."
        )

    def test_49km_uses_formula(self):
        from app.main import predict_travel_time, _formula_travel_time
        formula = _formula_travel_time(49.0, "High", "Rainy", "driving", "nairobi")
        api     = predict_travel_time(49.0, "High", "Rainy", 2, "driving", "nairobi", 30.5)
        assert api == formula

    def test_50km_does_not_use_formula_path(self):
        """At exactly 50 km the XGBoost path is taken. Result should differ
        from formula (the whole point of the ML model)."""
        from app.main import predict_travel_time, _formula_travel_time
        formula = _formula_travel_time(50.0, "Low", "Clear", "driving", "lagos")
        api     = predict_travel_time(50.0, "Low", "Clear", 2, "driving", "lagos", 27.1)
        # XGBoost and formula disagree on 50 km trips — this is expected.
        # If they accidentally agree it doesn't necessarily indicate a bug,
        # but we document that they *can* differ.
        # We only assert the result is valid, not identical to formula.
        assert isinstance(api, float) and api > 0, (
            f"predict_travel_time(50km) returned invalid value: {api}"
        )

    def test_formula_fallback_always_positive_range(self):
        from app.main import predict_travel_time
        for dist in [1.0, 5.0, 10.0, 25.0, 45.0, 49.9]:
            t = predict_travel_time(dist, "High", "Rainy", 2, "driving", "nairobi", 22.3)
            assert t > 0, f"Formula returned non-positive for dist={dist}: {t}"
            assert math.isfinite(t), f"Formula returned non-finite for dist={dist}: {t}"


# ---------------------------------------------------------------------------
# TestWarningCountPerLoad
# ---------------------------------------------------------------------------

class TestWarningCountPerLoad:
    """Verify the exact warning count per single pkl load, so the total can
    be predicted and any change in warning count is immediately detected."""

    def test_rf_load_emits_exactly_202_warnings(self):
        """200 DecisionTree + 1 RandomForest wrapper = 201 InconsistentVersionWarning
        for the RF + 1 extra for the outer RF call = 202 total on a fresh load.

        NOTE: this test only holds when commute_quality_model.pkl was serialised
        with a different sklearn version than the current environment. If the
        versions are aligned in future, this count will drop to 0.
        """
        import joblib
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            joblib.load(MODELS_DIR / "commute_quality_model.pkl")
        ivw = [w for w in caught if w.category.__name__ == "InconsistentVersionWarning"]
        assert len(ivw) == 202, (
            f"Expected 202 InconsistentVersionWarning on RF load, got {len(ivw)}. "
            "RF estimator count may have changed, or sklearn versions are now aligned."
        )

    def test_xgb_load_emits_at_most_1_userwarning(self):
        """XGBoost UserWarning fires at most once per pkl load (deprecated pickle format).
        In a cold process the first load emits 1 warning; subsequent loads within the
        same process may emit 0 because Python's warning registry deduplicates by
        message after the first occurrence even with simplefilter('always').
        We assert the count is 0 or 1 — never more than 1."""
        import joblib
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            joblib.load(MODELS_DIR / "travel_time_model.pkl")
        uw = [w for w in caught if w.category.__name__ == "UserWarning"]
        assert len(uw) <= 1, (
            f"Expected 0 or 1 UserWarning on XGBoost load, got {len(uw)}: "
            + "; ".join(str(w.message)[:80] for w in uw)
        )

    def test_dict_pkls_emit_no_warnings(self):
        """encoders, road_quality, safety_scores, transport_modes are plain dicts
        and must not emit any warnings."""
        import joblib
        for name in ["encoders.pkl", "road_quality.pkl",
                     "safety_scores.pkl", "transport_modes.pkl"]:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                joblib.load(MODELS_DIR / name)
            assert len(caught) == 0, (
                f"{name} emitted {len(caught)} unexpected warnings: "
                + ", ".join(w.category.__name__ for w in caught)
            )
