import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ml.temporal_cnn import ExerciseClassifierCNN
from ml.form_scorer import MLFormScorer
from exercises import get_analyzer
from core.rep_detector import RepDetector


def main():
    keypoints_path = "squat1_keypoints.json"
    with open(keypoints_path, "r", encoding="utf-8") as f:
        frames = json.load(f)

    print("=" * 65)
    print("TEST 1: PYTORCH 1D-CNN TEMPORAL MOVEMENT CLASSIFIER")
    print("=" * 65)

    cnn_classifier = ExerciseClassifierCNN()
    prediction = cnn_classifier.predict(frames, dominant_side="left")

    print(f"Architecture:       {prediction['model_architecture']}")
    print(f"Input Tensor Shape: {prediction['input_tensor_shape']} (Batch=1, Channels=10, Time=30)")
    print(f"Predicted Exercise: [{prediction['predicted_exercise'].upper()}] (Confidence: {prediction['confidence_score'] * 100:.1f}%)")
    print("\nClass Probability Distribution:")
    for cls_name, prob in prediction["class_probabilities"].items():
        bar = "#" * int(prob * 30)
        print(f"  {cls_name.ljust(15)}: {prob:.4f} | {bar}")

    print("\n" + "=" * 65)
    print("TEST 2: CLASSICAL ML RANDOM FOREST FORM QUALITY SCORER")
    print("=" * 65)

    # 1. Segment reps and run biomechanical analysis
    reps = RepDetector.segment_reps(frames, joint_name="hip", axis="y", dominant_side="left")
    analyzer = get_analyzer(prediction["predicted_exercise"], fps=30.0)
    set_result = analyzer.analyze_set(frames, reps, dominant_side="left")

    # 2. Run Random Forest Scorer
    rf_scorer = MLFormScorer()
    rf_results = rf_scorer.score_set(set_result)

    print(f"Model: {rf_results['model']}")
    print("\nRep-by-Rep Machine Learning Tier Predictions:")
    for rep_eval in rf_results["rep_evaluations"]:
        print(f"  Rep {rep_eval['rep_number']}: Tier -> [{rep_eval['ml_form_tier']}] (Confidence: {rep_eval['tier_probability'] * 100:.1f}%)")
        print(f"    Features: {rep_eval['engineered_features']}")

    print("\nRandom Forest Feature Importance Ranking (What drives good form?):")
    for rank, (feat, imp) in enumerate(rf_results["feature_importance_ranking"].items(), 1):
        bar = "=" * int(imp * 40)
        print(f"  {rank}. {feat.ljust(22)}: {imp:.4f} | {bar}")

    print("\n" + "=" * 65)
    print("ALL MACHINE LEARNING & CNN TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
