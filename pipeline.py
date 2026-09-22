import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Core Imports
from extract_pose import extract_keypoints
from core.rep_detector import RepDetector
from core.automotive_telemetry import AutomotiveKinematicsTelemetry
from core.video_annotator import VideoAnnotator

# Biomechanics & ML Imports
from exercises import get_analyzer
from ml.temporal_cnn import ExerciseClassifierCNN
from ml.form_scorer import MLFormScorer

# Coaching Imports
from coaching.groq_coach import GroqCoachEngine


class GymFormCheckerPipeline:
    """
    Unified Master Pipeline for AI Gym Form Analysis.
    Orchestrates Computer Vision, Biomechanical Rules, Automotive Telemetry,
    Machine Learning, RAG Knowledge Retrieval, and Groq LLM Coaching.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.cnn_classifier = ExerciseClassifierCNN()
        self.rf_scorer = MLFormScorer()
        self.auto_telemetry = AutomotiveKinematicsTelemetry(fps=fps)
        self.video_annotator = VideoAnnotator(fps=fps)
        self.groq_coach = GroqCoachEngine()

    def _determine_dominant_side(self, frames: list) -> str:
        """
        Auto-detects whether lifter is viewed from left or right side
        based on landmark visibility confidence.
        """
        left_vis, right_vis = 0.0, 0.0
        count = 0
        for f in frames[:60]:  # Sample first 60 frames
            if "left_hip" in f and "right_hip" in f:
                l_v = f["left_hip"].get("vis") or 0.5
                r_v = f["right_hip"].get("vis") or 0.5
                left_vis += l_v
                right_vis += r_v
                count += 1
        if count > 0 and right_vis > left_vis:
            return "right"
        return "left"

    def process_video(
        self,
        video_path: str,
        exercise_override: Optional[str] = None,
        generate_video: bool = False,
        output_json_path: Optional[str] = None,
        user_name: str = "Athlete"
    ) -> Dict[str, Any]:
        """
        Runs the complete end-to-end analysis on a workout video.
        """
        start_time = time.time()
        base_name = os.path.splitext(video_path)[0]
        cached_json = f"{base_name}_keypoints.json"

        print("\n" + "=" * 65)
        print("  AI GYM FORM CHECKER & TELEMETRY COACH - MASTER PIPELINE")
        print("=" * 65)
        print(f"Target Video: {video_path}")

        # -------------------------------------------------------------
        # 1. Computer Vision & Pose Extraction
        # -------------------------------------------------------------
        print("\n[1/6] Extracting skeletal landmarks via MediaPipe...")
        if os.path.exists(cached_json):
            print(f"      Loading cached keypoints from {cached_json}...")
            with open(cached_json, "r", encoding="utf-8") as f:
                frames_keypoints = json.load(f)
        else:
            print("      Processing video frames with PoseLandmarker...")
            frames_keypoints = extract_keypoints(video_path)
            with open(cached_json, "w", encoding="utf-8") as f:
                json.dump(frames_keypoints, f, indent=2)

        dominant_side = self._determine_dominant_side(frames_keypoints)
        print(f"      Extracted {len(frames_keypoints)} frames. Dominant side: [{dominant_side.upper()}]")

        # -------------------------------------------------------------
        # 2. PyTorch 1D-CNN Movement Recognition
        # -------------------------------------------------------------
        print("\n[2/6] Running 1D-CNN Temporal Movement Classifier...")
        cnn_pred = self.cnn_classifier.predict(frames_keypoints, dominant_side=dominant_side)
        detected_exercise = exercise_override or cnn_pred["predicted_exercise"]
        print(f"      Exercise Identified: [{detected_exercise.upper()}] (Confidence: {cnn_pred['confidence_score'] * 100:.1f}%)")

        # -------------------------------------------------------------
        # 3. Biomechanical Rule Engine & Kinematic Math
        # -------------------------------------------------------------
        print("\n[3/6] Segmenting reps & computing kinematic joint angles...")
        analyzer = get_analyzer(detected_exercise, fps=self.fps)
        tracking_joint, tracking_axis = analyzer.get_primary_tracking_joint()

        reps_indices = RepDetector.segment_reps(
            frames=frames_keypoints,
            joint_name=tracking_joint,
            axis=tracking_axis,
            dominant_side=dominant_side,
            prominence=0.04,
            min_distance_frames=10
        )
        set_result = analyzer.analyze_set(frames_keypoints, reps_indices, dominant_side=dominant_side)

        print(f"      Total Reps: {set_result.total_reps} | Successful: {set_result.successful_reps}")
        print(f"      Kinematic Form Score: {set_result.overall_score:.1f} / 100")
        if set_result.detected_faults_summary:
            print(f"      Faults Detected: {set_result.detected_faults_summary}")

        # -------------------------------------------------------------
        # 4. Automotive AI Telemetry & ADAS
        # -------------------------------------------------------------
        print("\n[4/6] Executing Automotive Telemetry (Kalman Filter + ADAS LDW + VBT)...")
        auto_telemetry_data = self.auto_telemetry.process_telemetry(
            frames=frames_keypoints,
            reps=reps_indices,
            dominant_side=dominant_side
        )
        vbt = auto_telemetry_data["velocity_telematics"]
        print(f"      Kalman Filtered Trajectories: {auto_telemetry_data['total_frames_analyzed']} frames")
        print(f"      Bar Path Lane Departures:    {auto_telemetry_data['lane_departure_events_count']} events")
        print(f"      Concentric Velocities:       {auto_telemetry_data['concentric_velocities_per_rep']}")
        print(f"      Set Velocity Loss:           {vbt['velocity_loss_pct']}% -> [{vbt['telematics_status']}]")

        # -------------------------------------------------------------
        # 5. Machine Learning (Random Forest Form Scorer)
        # -------------------------------------------------------------
        print("\n[5/6] Evaluating repetitions with Random Forest Classifier...")
        ml_eval = self.rf_scorer.score_set(set_result)
        tier_counts = {}
        for r in ml_eval["rep_evaluations"]:
            t = r["ml_form_tier"]
            tier_counts[t] = tier_counts.get(t, 0) + 1
        print(f"      ML Form Tiers: {tier_counts}")

        # -------------------------------------------------------------
        # 6. RAG Grounding & Groq LLM Coaching Report
        # -------------------------------------------------------------
        print("\n[6/6] Querying RAG Vector DB & generating Groq LLM coach debrief...")
        coach_output = self.groq_coach.generate_coach_report(
            set_result=set_result,
            user_name=user_name,
            skill_level="Intermediate"
        )
        print("      Coaching debrief synthesized successfully!")

        # -------------------------------------------------------------
        # 7. Video Visual Annotation
        # -------------------------------------------------------------
        annotated_video_path = None
        if generate_video and os.path.exists(video_path):
            annotated_video_path = f"{base_name}_annotated.mp4"
            print(f"\n[Optional] Rendering annotated video -> {annotated_video_path}...")
            self.video_annotator.render_annotated_video(
                video_path=video_path,
                output_path=annotated_video_path,
                frames_keypoints=frames_keypoints,
                set_result=set_result,
                dominant_side=dominant_side
            )
            print("      Video render complete.")

        total_runtime = time.time() - start_time

        # -------------------------------------------------------------
        # Compile Master Unified Payload
        # -------------------------------------------------------------
        master_payload = {
            "metadata": {
                "video_file": os.path.basename(video_path),
                "exercise": detected_exercise,
                "dominant_camera_side": dominant_side,
                "total_frames": len(frames_keypoints),
                "pipeline_runtime_seconds": round(total_runtime, 2),
                "annotated_video_path": annotated_video_path,
            },
            "kinematics": set_result.to_dict(),
            "automotive_telemetry": auto_telemetry_data,
            "machine_learning": {
                "1d_cnn_recognition": cnn_pred,
                "random_forest_scoring": ml_eval,
            },
            "rag_coaching": {
                "retrieved_protocols": coach_output.get("rag_protocols"),
                "coach_report_markdown": coach_output.get("coaching_report"),
            },
            "keypoints": frames_keypoints,
        }

        # Save output JSON
        save_path = output_json_path or f"{base_name}_analysis.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(master_payload, f, indent=2)

        print("\n" + "=" * 65)
        print(f"PIPELINE COMPLETED IN {total_runtime:.2f}s! Results saved -> {save_path}")
        print("=" * 65)
        print(coach_output["coaching_report"])

        return master_payload


def main():
    parser = argparse.ArgumentParser(description="AI Gym Form Checker & Telemetry Pipeline")
    parser.add_argument("--video", type=str, default="squat1.mp4", help="Path to input workout video")
    parser.add_argument("--exercise", type=str, default=None, help="Override exercise (squat, deadlift, bench_press, etc.)")
    parser.add_argument("--no-video", action="store_true", help="Skip rendering annotated video")
    parser.add_argument("--output", type=str, default=None, help="Output JSON results path")
    parser.add_argument("--user", type=str, default="Lifter", help="User / Athlete name")
    args = parser.parse_args()

    pipeline = GymFormCheckerPipeline()
    pipeline.process_video(
        video_path=args.video,
        exercise_override=args.exercise,
        generate_video=not args.no_video,
        output_json_path=args.output,
        user_name=args.user
    )


if __name__ == "__main__":
    main()
