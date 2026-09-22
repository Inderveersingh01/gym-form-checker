import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.rep_detector import RepDetector
from core.automotive_telemetry import AutomotiveKinematicsTelemetry


def main():
    keypoints_path = "squat1_keypoints.json"
    with open(keypoints_path, "r", encoding="utf-8") as f:
        frames = json.load(f)

    print("=" * 60)
    print("AUTOMOTIVE AI & TELEMATICS TEST RUNNER")
    print("=" * 60)

    # 1. Segment reps
    reps = RepDetector.segment_reps(
        frames=frames,
        joint_name="hip",
        axis="y",
        dominant_side="left",
        prominence=0.04,
        min_distance_frames=10
    )
    print(f"Loaded {len(frames)} frames. Detected {len(reps)} reps.")

    # 2. Run Automotive Telemetry Engine
    auto_telemetry = AutomotiveKinematicsTelemetry(fps=30.0, lane_tolerance_pct=0.05)
    results = auto_telemetry.process_telemetry(frames, reps, dominant_side="left")

    print("\n--- 1. KALMAN FILTER STATE-SPACE TRACKING ---")
    print(f"Successfully processed {results['total_frames_analyzed']} frames through 1D Kalman Filter.")
    print("Joint noise attenuated and instantaneous velocity vectors estimated.")

    print("\n--- 2. ADAS: BAR-PATH LANE DEPARTURE WARNING (LDW) ---")
    print(f"Lane Departure Events Detected: {results['lane_departure_events_count']}")
    if results["lane_departures_sample"]:
        print("Sample Lane Departure Alerts:")
        for dep in results["lane_departures_sample"]:
            print(f"  * Frame {dep['frame']}: Drift {dep['drift_pct']}% -> {dep['warning']}")
    else:
        print("  Bar remained inside the midfoot stability envelope throughout the set.")

    print("\n--- 3. VEHICLE-STYLE TELEMATICS: VELOCITY-BASED TRAINING (VBT) ---")
    vel_data = results["velocity_telematics"]
    print(f"Concentric Velocities per Rep (m/s proxy): {results['concentric_velocities_per_rep']}")
    print(f"Fastest Rep Velocity: {vel_data['fastest_rep_concentric_vel']}")
    print(f"Final Rep Velocity:   {vel_data['final_rep_concentric_vel']}")
    print(f"Set Velocity Loss:    {vel_data['velocity_loss_pct']}% (Threshold: {vel_data['fatigue_threshold_pct']}%)")
    print(f"Fatigue Status:       [{vel_data['telematics_status']}]")

    print("\n" + "=" * 60)
    print("AUTOMOTIVE AI PIPELINE EXECUTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
