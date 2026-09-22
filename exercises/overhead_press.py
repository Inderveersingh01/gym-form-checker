from typing import Any, Dict, List, Tuple
from exercises.base import BaseExerciseAnalyzer, Fault, FaultSeverity, Kinematics, RepAnalysis


class OverheadPressAnalyzer(BaseExerciseAnalyzer):
    """
    Biomechanical rule engine for Overhead Press / Military Press.
    Analyzes overhead lockout, excessive lumbar hyperextension, and rack position.
    """

    def __init__(
        self,
        fps: float = 30.0,
        max_torso_backward_lean: float = 16.0,
        min_lockout_elbow_angle: float = 165.0,
    ):
        super().__init__(exercise_name="overhead_press", fps=fps)
        self.max_torso_backward_lean = max_torso_backward_lean
        self.min_lockout_elbow_angle = min_lockout_elbow_angle

    def get_primary_tracking_joint(self) -> Tuple[str, str]:
        # Wrist y dictates rep cycle (min y = top lockout overhead)
        return ("wrist", "y")

    def analyze_rep(
        self,
        frames: List[Dict[str, Any]],
        rep_info: Dict[str, int],
        dominant_side: str = "left"
    ) -> RepAnalysis:
        start_idx = rep_info["start_idx"]
        overhead_idx = rep_info["inflection_idx"]  # Top lockout overhead
        end_idx = rep_info["end_idx"]
        rep_num = rep_info["rep_number"]

        start_frame_num = frames[start_idx]["frame"]
        overhead_frame_num = frames[overhead_idx]["frame"]
        end_frame_num = frames[end_idx]["frame"]

        duration_sec = (end_frame_num - start_frame_num) / self.fps
        concentric_sec = (overhead_frame_num - start_frame_num) / self.fps
        eccentric_sec = (end_frame_num - overhead_frame_num) / self.fps

        faults: List[Fault] = []
        metrics: Dict[str, Any] = {}

        # 1. Overhead Lockout Phase Analysis
        overhead_frame = frames[overhead_idx]
        shoulder = Kinematics.get_joint(overhead_frame, "shoulder", dominant_side)
        elbow = Kinematics.get_joint(overhead_frame, "elbow", dominant_side)
        wrist = Kinematics.get_joint(overhead_frame, "wrist", dominant_side)
        hip = Kinematics.get_joint(overhead_frame, "hip", dominant_side)

        # Overhead elbow extension angle
        elbow_lockout = Kinematics.calculate_angle_3p(shoulder, elbow, wrist)
        metrics["elbow_lockout_angle"] = elbow_lockout

        if elbow_lockout < self.min_lockout_elbow_angle:
            faults.append(
                Fault(
                    code="INCOMPLETE_OVERHEAD_LOCKOUT",
                    name="Soft Overhead Lockout",
                    severity=FaultSeverity.MEDIUM,
                    description=f"Elbows did not fully lock out overhead ({elbow_lockout:.1f} deg, target >= {self.min_lockout_elbow_angle:.0f} deg). Press the ceiling away.",
                    rep_number=rep_num,
                    frame=overhead_frame_num,
                    metric_value=elbow_lockout,
                    target_threshold=self.min_lockout_elbow_angle,
                    phase="lockout",
                )
            )

        # 2. Lumbar Hyperextension / Excessive Backward Lean
        torso_lean = Kinematics.calculate_angle_with_vertical(hip, shoulder)
        metrics["torso_lean_at_lockout"] = torso_lean

        # Backward lean check (shoulder positioned behind hip on horizontal plane)
        leaning_back = shoulder["x"] < hip["x"] if dominant_side == "left" else shoulder["x"] > hip["x"]
        if leaning_back and torso_lean > self.max_torso_backward_lean:
            faults.append(
                Fault(
                    code="LUMBAR_HYPEREXTENSION",
                    name="Excessive Lower Back Arching",
                    severity=FaultSeverity.HIGH,
                    description=f"Leaning backward excessively ({torso_lean:.1f} deg from vertical). Squeeze glutes and brace core to protect the spine.",
                    rep_number=rep_num,
                    frame=overhead_frame_num,
                    metric_value=torso_lean,
                    target_threshold=self.max_torso_backward_lean,
                    phase="lockout",
                )
            )

        # 3. Stacked Joint Alignment (Wrist over Shoulder)
        wrist_shoulder_drift = abs(wrist["x"] - shoulder["x"])
        metrics["wrist_shoulder_drift"] = wrist_shoulder_drift
        if wrist_shoulder_drift > 0.08:
            faults.append(
                Fault(
                    code="UNSTACKED_OVERHEAD",
                    name="Bar Forward of Shoulders",
                    severity=FaultSeverity.MEDIUM,
                    description="Bar ended forward of shoulders instead of directly overhead in line with spine.",
                    rep_number=rep_num,
                    frame=overhead_frame_num,
                    metric_value=wrist_shoulder_drift,
                    target_threshold=0.08,
                    phase="lockout",
                )
            )

        passed = not any(f.severity == FaultSeverity.HIGH for f in faults)

        return RepAnalysis(
            rep_number=rep_num,
            start_frame=start_frame_num,
            inflection_frame=overhead_frame_num,
            end_frame=end_frame_num,
            duration_seconds=duration_sec,
            eccentric_seconds=eccentric_sec,
            concentric_seconds=concentric_sec,
            passed=passed,
            faults=faults,
            metrics=metrics,
        )
