import json
import os
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.rep_detector import RepDetector
from exercises import get_analyzer
from core.video_annotator import VideoAnnotator


def main():
    video_path = "squat1.mp4"
    keypoints_path = "squat1_keypoints.json"
    output_path = "squat1_annotated.mp4"

    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found!")
        return

    print("=" * 60)
    print("STEP 1: LOADING DATA & RUNNING KINEMATICS")
    print("=" * 60)

    with open(keypoints_path, "r", encoding="utf-8") as f:
        frames = json.load(f)

    # 1. Segment reps
    reps = RepDetector.segment_reps(frames, joint_name="hip", axis="y", dominant_side="left")
    analyzer = get_analyzer("squat", fps=30.0)
    set_result = analyzer.analyze_set(frames, reps, dominant_side="left")

    print(f"Loaded {len(frames)} keypoint frames.")
    print(f"Detected {set_result.total_reps} reps. Overall Score: {set_result.overall_score:.1f}/100")

    print("\n" + "=" * 60)
    print("STEP 2: RENDERING ANNOTATED VIDEO (OPENCV)")
    print("=" * 60)
    print(f"Input Video:  {video_path}")
    print(f"Output Video: {output_path}")

    annotator = VideoAnnotator(fps=30.0)
    t0 = time.time()
    annotator.render_annotated_video(
        video_path=video_path,
        output_path=output_path,
        frames_keypoints=frames,
        set_result=set_result,
        dominant_side="left"
    )
    render_time = time.time() - t0

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n[SUCCESS] Rendered annotated video in {render_time:.2f}s!")
        print(f"Output File: {output_path} ({size_mb:.2f} MB)")
        print("Visual elements rendered:")
        print("  - Neon green skeletal bone connections and joint circles")
        print("  - Live knee angle and torso angle degrees")
        print("  - Top-left HUD dashboard with real-time rep counter")
        print("  - Color-coded badges: [DEPTH: HIT] (Green), [FORWARD LEAN] (Red)")
    else:
        print("Error: Annotated video was not produced.")


if __name__ == "__main__":
    main()
