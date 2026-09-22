from typing import Any, Dict, List, Tuple
from exercises.base import BaseExerciseAnalyzer, Fault, FaultSeverity, Kinematics, RepAnalysis


class SquatAnalyzer(BaseExerciseAnalyzer):
    """
    Biomechanical rule engine for Squats (Back Squat, Front Squat, Goblet Squat).
    Optimized for side view and 45-degree angle.
    """

    def __init__(
        self,
        fps: float = 30.0,
        depth_threshold_angle: float = 90.0,      # Max knee angle at bottom for parallel depth (degrees)
        max_torso_lean_angle: float = 40.0,       # Max torso angle from vertical (degrees)
        min_eccentric_time: float = 0.8,          # Prevent dive-bombing squats
    ):
        super().__init__(exercise_name="squat", fps=fps)
        self.depth_threshold_angle = depth_threshold_angle
        self.max_torso_lean_angle = max_torso_lean_angle
        self.min_eccentric_time = min_eccentric_time

    def get_primary_tracking_joint(self) -> Tuple[str, str]:
        # Hip vertical position dictates squat rep cycle (max y = bottom of squat)
        return ("hip", "y")

    def analyze_rep(
        self,
        frames: List[Dict[str, Any]],
        rep_info: Dict[str, int],
        dominant_side: str = "left"
    ) -> RepAnalysis:
        start_idx = rep_info["start_idx"]
        bottom_idx = rep_info["inflection_idx"]
        end_idx = rep_info["end_idx"]
        rep_num = rep_info["rep_number"]

        start_frame_num = frames[start_idx]["frame"]
        bottom_frame_num = frames[bottom_idx]["frame"]
        end_frame_num = frames[end_idx]["frame"]

        duration_sec = (end_frame_num - start_frame_num) / self.fps
        eccentric_sec = (bottom_frame_num - start_frame_num) / self.fps
        concentric_sec = (end_frame_num - bottom_frame_num) / self.fps

        faults: List[Fault] = []
        metrics: Dict[str, Any] = {}

        # 1. Depth Evaluation at the Bottom (Inflection Point)
        bottom_frame = frames[bottom_idx]
        hip = Kinematics.get_joint(bottom_frame, "hip", dominant_side)
        knee = Kinematics.get_joint(bottom_frame, "knee", dominant_side)
        ankle = Kinematics.get_joint(bottom_frame, "ankle", dominant_side)
        shoulder = Kinematics.get_joint(bottom_frame, "shoulder", dominant_side)

        knee_angle_at_bottom = Kinematics.calculate_angle_3p(hip, knee, ankle)
        # Normalized coordinate difference: in screen coords, larger y is lower
        hip_knee_y_diff = hip["y"] - knee["y"]

        metrics["knee_angle_at_bottom"] = knee_angle_at_bottom
        metrics["hip_knee_y_diff"] = hip_knee_y_diff

        # Check Depth:
        # Pass criteria: knee angle <= threshold OR hip crease at or below knee (hip_y >= knee_y)
        if knee_angle_at_bottom > self.depth_threshold_angle and hip_knee_y_diff < 0.0:
            faults.append(
                Fault(
                    code="SHALLOW_DEPTH",
                    name="Shallow Squat Depth",
                    severity=FaultSeverity.HIGH,
                    description=f"Knee angle reached {knee_angle_at_bottom:.1f} deg (target <= {self.depth_threshold_angle:.0f} deg). Hip crease remained above knee joint.",
                    rep_number=rep_num,
                    frame=bottom_frame_num,
                    metric_value=knee_angle_at_bottom,
                    target_threshold=self.depth_threshold_angle,
                    phase="bottom",
                )
            )
            metrics["depth_status"] = "SHALLOW"
        elif knee_angle_at_bottom <= 75.0:
            metrics["depth_status"] = "DEEP"
        else:
            metrics["depth_status"] = "PARALLEL"

        # 2. Torso Forward Lean at Bottom
        torso_angle = Kinematics.calculate_angle_with_vertical(hip, shoulder)
        metrics["torso_lean_at_bottom"] = torso_angle

        if torso_angle > self.max_torso_lean_angle:
            faults.append(
                Fault(
                    code="EXCESSIVE_FORWARD_LEAN",
                    name="Excessive Forward Lean",
                    severity=FaultSeverity.MEDIUM,
                    description=f"Torso leaned {torso_angle:.1f} deg from vertical (target <= {self.max_torso_lean_angle:.0f} deg). Watch for good-morning squat pattern.",
                    rep_number=rep_num,
                    frame=bottom_frame_num,
                    metric_value=torso_angle,
                    target_threshold=self.max_torso_lean_angle,
                    phase="bottom",
                )
            )

        # 3. Descent Control (Dive-Bombing / Rapid Descent)
        if eccentric_sec < self.min_eccentric_time:
            faults.append(
                Fault(
                    code="RAPID_DESCENT",
                    name="Uncontrolled Descent",
                    severity=FaultSeverity.LOW,
                    description=f"Descent duration was {eccentric_sec:.2f}s (recommended >= {self.min_eccentric_time:.1f}s). Control the eccentric phase for stability.",
                    rep_number=rep_num,
                    frame=bottom_frame_num,
                    metric_value=eccentric_sec,
                    target_threshold=self.min_eccentric_time,
                    phase="eccentric",
                )
            )

        # 4. Lockout at Rep Completion (Knee fully extended at start and end)
        end_frame = frames[end_idx]
        end_hip = Kinematics.get_joint(end_frame, "hip", dominant_side)
        end_knee = Kinematics.get_joint(end_frame, "knee", dominant_side)
        end_ankle = Kinematics.get_joint(end_frame, "ankle", dominant_side)
        lockout_knee_angle = Kinematics.calculate_angle_3p(end_hip, end_knee, end_ankle)
        metrics["lockout_knee_angle"] = lockout_knee_angle

        if lockout_knee_angle < 160.0:
            faults.append(
                Fault(
                    code="INCOMPLETE_LOCKOUT",
                    name="Incomplete Lockout",
                    severity=FaultSeverity.LOW,
                    description=f"Knees did not reach full extension at the top ({lockout_knee_angle:.1f} deg, target >= 160 deg). Stand tall between reps.",
                    rep_number=rep_num,
                    frame=end_frame_num,
                    metric_value=lockout_knee_angle,
                    target_threshold=160.0,
                    phase="lockout",
                )
            )

        passed = not any(f.severity == FaultSeverity.HIGH for f in faults)

        return RepAnalysis(
            rep_number=rep_num,
            start_frame=start_frame_num,
            inflection_frame=bottom_frame_num,
            end_frame=end_frame_num,
            duration_seconds=duration_sec,
            eccentric_seconds=eccentric_sec,
            concentric_seconds=concentric_sec,
            passed=passed,
            faults=faults,
            metrics=metrics,
        )
