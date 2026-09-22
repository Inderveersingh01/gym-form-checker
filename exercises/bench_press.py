from typing import Any, Dict, List, Tuple
from exercises.base import BaseExerciseAnalyzer, Fault, FaultSeverity, Kinematics, RepAnalysis


class BenchPressAnalyzer(BaseExerciseAnalyzer):
    """
    Biomechanical rule engine for Bench Press and Push-ups.
    Analyzes elbow flare, range of motion (chest touch), and lockout.
    """

    def __init__(
        self,
        fps: float = 30.0,
        max_elbow_flare_angle: float = 75.0,    # Max angle between upper arm and torso (degrees)
        min_elbow_lockout_angle: float = 160.0, # Full elbow extension at lockout
        min_eccentric_time: float = 0.8,
    ):
        super().__init__(exercise_name="bench_press", fps=fps)
        self.max_elbow_flare_angle = max_elbow_flare_angle
        self.min_elbow_lockout_angle = min_elbow_lockout_angle
        self.min_eccentric_time = min_eccentric_time

    def get_primary_tracking_joint(self) -> Tuple[str, str]:
        # Wrist vertical trajectory dictates rep cycle (max y = chest touch in screen space)
        return ("wrist", "y")

    def analyze_rep(
        self,
        frames: List[Dict[str, Any]],
        rep_info: Dict[str, int],
        dominant_side: str = "left"
    ) -> RepAnalysis:
        start_idx = rep_info["start_idx"]
        chest_idx = rep_info["inflection_idx"]  # Lowest point of bar / chest touch
        end_idx = rep_info["end_idx"]
        rep_num = rep_info["rep_number"]

        start_frame_num = frames[start_idx]["frame"]
        chest_frame_num = frames[chest_idx]["frame"]
        end_frame_num = frames[end_idx]["frame"]

        duration_sec = (end_frame_num - start_frame_num) / self.fps
        eccentric_sec = (chest_frame_num - start_frame_num) / self.fps
        concentric_sec = (end_frame_num - chest_frame_num) / self.fps

        faults: List[Fault] = []
        metrics: Dict[str, Any] = {}

        # 1. Chest Touch / Bottom Phase Analysis
        bottom_frame = frames[chest_idx]
        shoulder = Kinematics.get_joint(bottom_frame, "shoulder", dominant_side)
        elbow = Kinematics.get_joint(bottom_frame, "elbow", dominant_side)
        wrist = Kinematics.get_joint(bottom_frame, "wrist", dominant_side)
        hip = Kinematics.get_joint(bottom_frame, "hip", dominant_side)

        # Elbow flexion angle at chest (shoulder-elbow-wrist)
        elbow_angle_at_bottom = Kinematics.calculate_angle_3p(shoulder, elbow, wrist)
        metrics["elbow_angle_at_bottom"] = elbow_angle_at_bottom

        # Elbow flare angle (torso-to-upper-arm angle: hip-shoulder-elbow)
        elbow_flare_angle = Kinematics.calculate_angle_3p(hip, shoulder, elbow)
        metrics["elbow_flare_angle"] = elbow_flare_angle

        if elbow_flare_angle > self.max_elbow_flare_angle:
            faults.append(
                Fault(
                    code="EXCESSIVE_ELBOW_FLARE",
                    name="Excessive Elbow Flare",
                    severity=FaultSeverity.HIGH,
                    description=f"Elbows flared to {elbow_flare_angle:.1f} deg relative to torso (target <= {self.max_elbow_flare_angle:.0f} deg). Tuck elbows to 45-60 deg to protect rotator cuffs.",
                    rep_number=rep_num,
                    frame=chest_frame_num,
                    metric_value=elbow_flare_angle,
                    target_threshold=self.max_elbow_flare_angle,
                    phase="bottom",
                )
            )

        # Depth check: elbow should achieve ~90 degrees or lower
        if elbow_angle_at_bottom > 95.0:
            faults.append(
                Fault(
                    code="INCOMPLETE_ROM",
                    name="Incomplete Range of Motion",
                    severity=FaultSeverity.MEDIUM,
                    description=f"Bar stopped high ({elbow_angle_at_bottom:.1f} deg elbow flexion). Touch chest with control.",
                    rep_number=rep_num,
                    frame=chest_frame_num,
                    metric_value=elbow_angle_at_bottom,
                    target_threshold=90.0,
                    phase="bottom",
                )
            )

        # 2. Descent Control / Bouncing
        if eccentric_sec < self.min_eccentric_time:
            faults.append(
                Fault(
                    code="BOUNCING_BAR",
                    name="Bouncing / Fast Rebound",
                    severity=FaultSeverity.LOW,
                    description=f"Descent was too fast ({eccentric_sec:.2f}s). Lower the bar under strict control.",
                    rep_number=rep_num,
                    frame=chest_frame_num,
                    metric_value=eccentric_sec,
                    target_threshold=self.min_eccentric_time,
                    phase="eccentric",
                )
            )

        # 3. Lockout at Top
        end_frame = frames[end_idx]
        end_shoulder = Kinematics.get_joint(end_frame, "shoulder", dominant_side)
        end_elbow = Kinematics.get_joint(end_frame, "elbow", dominant_side)
        end_wrist = Kinematics.get_joint(end_frame, "wrist", dominant_side)
        lockout_elbow_angle = Kinematics.calculate_angle_3p(end_shoulder, end_elbow, end_wrist)
        metrics["lockout_elbow_angle"] = lockout_elbow_angle

        if lockout_elbow_angle < self.min_elbow_lockout_angle:
            faults.append(
                Fault(
                    code="INCOMPLETE_LOCKOUT",
                    name="Incomplete Elbow Lockout",
                    severity=FaultSeverity.LOW,
                    description=f"Elbows did not fully extend at lockout ({lockout_elbow_angle:.1f} deg, target >= {self.min_elbow_lockout_angle:.0f} deg).",
                    rep_number=rep_num,
                    frame=end_frame_num,
                    metric_value=lockout_elbow_angle,
                    target_threshold=self.min_elbow_lockout_angle,
                    phase="lockout",
                )
            )

        passed = not any(f.severity == FaultSeverity.HIGH for f in faults)

        return RepAnalysis(
            rep_number=rep_num,
            start_frame=start_frame_num,
            inflection_frame=chest_frame_num,
            end_frame=end_frame_num,
            duration_seconds=duration_sec,
            eccentric_seconds=eccentric_sec,
            concentric_seconds=concentric_sec,
            passed=passed,
            faults=faults,
            metrics=metrics,
        )
