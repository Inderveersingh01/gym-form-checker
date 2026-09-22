import numpy as np
from typing import Any, Dict, List
from sklearn.ensemble import RandomForestClassifier
from exercises.base import RepAnalysis, SetAnalysisResult


class KinematicFeatureExtractor:
    """
    Classical ML Feature Engineering:
    Transforms raw rep trajectories into dense tabular kinematic feature vectors.
    """

    FEATURE_NAMES = [
        "min_knee_angle",
        "torso_lean_at_bottom",
        "eccentric_seconds",
        "concentric_seconds",
        "tempo_ratio",
        "lockout_knee_angle",
        "fault_count",
    ]

    @classmethod
    def extract_features(cls, rep: RepAnalysis) -> np.ndarray:
        """
        Extracts fixed-length 7D numerical feature vector from a single rep.
        """
        m = rep.metrics
        min_knee = float(m.get("knee_angle_at_bottom", 90.0))
        torso_lean = float(m.get("torso_lean_at_bottom", 30.0))
        ecc_time = max(0.1, rep.eccentric_seconds)
        conc_time = max(0.1, rep.concentric_seconds)
        tempo_ratio = ecc_time / conc_time
        lockout = float(m.get("lockout_knee_angle", 175.0))
        fault_cnt = len(rep.faults)

        return np.array([
            min_knee,
            torso_lean,
            ecc_time,
            conc_time,
            tempo_ratio,
            lockout,
            fault_cnt,
        ], dtype=np.float32)


class MLFormScorer:
    """
    Classical Machine Learning Form Scorer using Random Forest Ensemble.
    Classifies repetitions into Form Quality Tiers and calculates
    feature importance rankings.
    """

    CLASSES = ["NEEDS_CORRECTION", "ACCEPTABLE", "EXCELLENT"]

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        self._initialize_and_fit_reference_model()

    def _initialize_and_fit_reference_model(self):
        """
        Fits Random Forest on a foundational synthetic dataset of
        biomechanical distributions based on NSCA exercise standards.
        """
        # Features: [min_knee, torso_lean, ecc_time, conc_time, tempo_ratio, lockout, fault_cnt]
        X = [
            # EXCELLENT (Label 2)
            [65.0, 28.0, 2.0, 1.2, 1.66, 175.0, 0],
            [70.0, 32.0, 1.8, 1.1, 1.63, 178.0, 0],
            [68.0, 30.0, 2.2, 1.3, 1.69, 174.0, 0],
            [75.0, 25.0, 1.9, 1.2, 1.58, 176.0, 0],

            # ACCEPTABLE (Label 1)
            [85.0, 42.0, 1.4, 1.5, 0.93, 170.0, 1],
            [88.0, 44.0, 1.2, 1.6, 0.75, 168.0, 1],
            [55.0, 48.0, 1.0, 2.0, 0.50, 172.0, 1],
            [50.0, 52.0, 1.6, 2.2, 0.72, 175.0, 1],

            # NEEDS_CORRECTION (Label 0)
            [105.0, 60.0, 0.5, 0.6, 0.83, 150.0, 3],
            [110.0, 65.0, 0.4, 0.5, 0.80, 145.0, 2],
            [95.0,  58.0, 0.6, 0.8, 0.75, 140.0, 2],
            [80.0,  62.0, 0.5, 1.0, 0.50, 130.0, 2],
        ]
        y = [2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0]
        self.model.fit(X, y)

    def score_set(self, set_result: SetAnalysisResult) -> Dict[str, Any]:
        """
        Applies Random Forest classifier across all reps in a workout set.
        """
        rep_evaluations = []

        for rep in set_result.reps:
            features = KinematicFeatureExtractor.extract_features(rep)
            probs = self.model.predict_proba([features])[0]
            pred_idx = int(np.argmax(probs))
            predicted_tier = self.CLASSES[pred_idx]
            quality_prob = float(probs[pred_idx])

            rep_evaluations.append({
                "rep_number": rep.rep_number,
                "ml_form_tier": predicted_tier,
                "tier_probability": round(quality_prob, 3),
                "engineered_features": {
                    name: round(float(val), 2)
                    for name, val in zip(KinematicFeatureExtractor.FEATURE_NAMES, features)
                },
            })

        # Feature Importance Analysis
        feature_importance = {
            name: round(float(imp), 4)
            for name, imp in zip(KinematicFeatureExtractor.FEATURE_NAMES, self.model.feature_importances_)
        }
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

        return {
            "model": "RandomForestClassifier (50 Estimators, max_depth=4)",
            "rep_evaluations": rep_evaluations,
            "feature_importance_ranking": sorted_importance,
        }
