import json
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from exercises import get_analyzer
from core.rep_detector import RepDetector
from coaching.groq_coach import GroqCoachEngine

def main():
    keypoints_path = "squat1_keypoints.json"
    with open(keypoints_path, "r", encoding="utf-8") as f:
        frames = json.load(f)

    print("=" * 60)
    print("STEP 1: KINEMATIC REP DETECTION & BIOMECHANICS")
    print("=" * 60)
    reps = RepDetector.segment_reps(
        frames=frames,
        joint_name="hip",
        axis="y",
        dominant_side="left",
        prominence=0.04,
        min_distance_frames=10
    )
    print(f"Segmented {len(reps)} repetitions from {len(frames)} frames.")

    analyzer = get_analyzer("squat", fps=30.0)
    set_result = analyzer.analyze_set(frames, reps, dominant_side="left")
    print(f"Overall Form Score: {set_result.overall_score:.1f} / 100")
    print(f"Detected Faults: {set_result.detected_faults_summary}")

    print("\n" + "=" * 60)
    print("STEP 2: RAG RETRIEVAL (STRENGTH SCIENCE LITERATURE)")
    print("=" * 60)
    coach = GroqCoachEngine()
    rag_retrieval = coach.retriever.retrieve_corrections_for_faults(
        exercise=set_result.exercise,
        faults=[f for r in set_result.reps for f in r.faults]
    )

    for p in rag_retrieval.get("retrieved_protocols", []):
        print(f"\n[RETRIEVED KNOWLEDGE CARD]: {p['fault_name']}")
        print(f"  Source: {p['source']}")
        print(f"  Biomechanical Root Cause: {p['biomechanical_root_cause']}")
        print("  Verified Cues:")
        for cue in p["verbal_cues"]:
            print(f"    * \"{cue}\"")
        print("  Targeted Corrective Drills:")
        for drill in p["corrective_drills"]:
            print(f"    * {drill['drill']} ({drill['prescription']}): {drill['coaching_points']}")

    print("\n" + "=" * 60)
    print("STEP 3: GROQ LLM GROUNDED COACHING REPORT")
    print("=" * 60)
    output = coach.generate_coach_report(set_result, user_name="Alex", skill_level="Intermediate")
    print(output["coaching_report"])

if __name__ == "__main__":
    main()
