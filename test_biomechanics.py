import json
import pprint
from exercises import get_analyzer
from core.rep_detector import RepDetector

def main():
    keypoints_path = "squat1_keypoints.json"
    with open(keypoints_path) as f:
        frames = json.load(f)

    print(f"Loaded {len(frames)} frames from {keypoints_path}")

    # 1. Segment reps using universal RepDetector
    reps = RepDetector.segment_reps(
        frames=frames,
        joint_name="hip",
        axis="y",
        dominant_side="left",
        invert=False,
        prominence=0.04,
        min_distance_frames=10
    )

    print(f"\nDetected {len(reps)} repetitions:")
    for r in reps:
        print(f"  Rep {r['rep_number']}: Start frame {frames[r['start_idx']]['frame']} -> Bottom frame {frames[r['inflection_idx']]['frame']} -> End frame {frames[r['end_idx']]['frame']}")

    # 2. Run Squat Biomechanical Engine
    analyzer = get_analyzer("squat", fps=30.0)
    result = analyzer.analyze_set(frames, reps, dominant_side="left")

    print(f"\n--- SET ANALYSIS RESULT ---")
    print(f"Exercise: {result.exercise}")
    print(f"Total Reps: {result.total_reps} | Successful: {result.successful_reps}")
    print(f"Overall Form Score: {result.overall_score:.1f} / 100")
    print(f"Detected Faults Summary: {result.detected_faults_summary}")

    print("\n--- REP-BY-REP BREAKDOWN ---")
    for rep in result.reps:
        status = "PASSED" if rep.passed else "FAILED"
        print(f"\nRep {rep.rep_number} [{status}]:")
        print(f"  Duration: {rep.duration_seconds:.2f}s (Eccentric: {rep.eccentric_seconds:.2f}s, Concentric: {rep.concentric_seconds:.2f}s)")
        print(f"  Knee Angle at Bottom: {rep.metrics.get('knee_angle_at_bottom', 0):.1f} deg ({rep.metrics.get('depth_status')})")
        print(f"  Torso Lean at Bottom: {rep.metrics.get('torso_lean_at_bottom', 0):.1f} deg")
        if rep.faults:
            print("  Faults Detected:")
            for f in rep.faults:
                print(f"    - [{f.severity.value.upper()}] {f.name}: {f.description}")
        else:
            print("  Clean rep! No faults detected.")

    print("\n--- JSON OUTPUT TEST ---")
    json_output = result.to_dict()
    print("to_dict() successfully produced valid JSON-serializable output:")
    print(json.dumps(json_output, indent=2)[:500] + "\n... [truncated]")

if __name__ == "__main__":
    main()
