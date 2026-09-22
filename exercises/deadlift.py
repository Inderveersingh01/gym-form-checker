from typing import Any, Dict, List, Tuple
from exercises.base import BaseExerciseAnalyzer, Fault, FaultSeverity, Kinematics, RepAnalysis


class DeadliftAnalyzer(BaseExerciseAnalyzer):
    """
    Biomechanical rule engine for Deadlifts (Conventional, Sumo, Romanian Deadlift).
    Optimized for side view.
    """

    def __init__(
        self,
        fps: float = 30.0,
        max_torso_angle_deviation: float = 12.0,   # Max torso angle change (rounding) during first half of pull
        max_bar_horizontal_drift: float = 0.08,    # Max horizontal drift from ankle normalized to frame width
        max_hyperextension_angle: float = 8.0,     # Max backward lean at lockout
    ):
        super().__init__(exercise_name="deadlift", fps=fps)
        self.max_torso_angle_deviation = max_torso_angle_deviation
        self.max_bar_horizontal_drift = max_bar_horizontal_drift
        self.max_hyperextension_angle = max_hyperextension_angle

    def get_primary_tracking_joint(self) -> Tuple[str, str]:
        # For deadlift, hip or wrist vertical trajectory dictates rep cycle (top of deadlift = min y)
        return ("hip", "y")

    def analyze_rep(
        self,
        frames: List[Dict[str, Any]],
        rep_info: Dict[str, int],
        dominant_side: str = "left"
    ) -> RepAnalysis:
        start_idx = rep_info["start_idx"]
        lockout_idx = rep_info["inflection_idx"]  # Standing tall at top of deadlift
        end_idx = rep_info["end_idx"]
        rep_num = rep_info["rep_number"]

        start_frame_num = frames[start_idx]["frame"]
        lockout_frame_num = frames[lockout_idx]["frame"]
        end_frame_num = frames[end_idx]["frame"]

        duration_sec = (end_frame_num - start_frame_num) / self.fps
        concentric_sec = (lockout_frame_num - start_frame_num) / self.fps
        eccentric_sec = (end_frame_num - lockout_frame_num) / self.fps

        faults: List[Fault] = []
        metrics: Dict[str, Any] = {}

        # 1. Starting Setup Check (Floor Position)
        start_frame = frames[start_idx]
        start_hip = Kinematics.get_joint(start_frame, "hip", dominant_side)
        start_knee = Kinematics.get_joint(start_frame, "knee", dominant_side)
        start_shoulder = Kinematics.get_joint(start_frame, "shoulder", dominant_side)

        # In screen coords, larger y is lower
        # Ideal: knee_y > hip_y > shoulder_y
        if start_hip["y"] >= start_knee["y"]:
            faults.append(
                Fault(
                    code="SQUATTING_THE_DEADLIFT",
                    name="Hips Too Low at Setup",
                    severity=FaultSeverity.MEDIUM,
                    description="Hips started below knee height. Avoid turning the deadlift setup into a squat.",
                    rep_number=rep_num,
                    frame=start_frame_num,
                    metric_value=start_hip["y"],
                    target_threshold=start_knee["y"],
                    phase="setup",
                )
            )
        elif start_hip["y"] <= start_shoulder["y"]:
            faults.append(
                Fault(
                    code="HIPS_TOO_HIGH_AT_START",
                    name="Hips Too High at Setup",
                    severity=FaultSeverity.MEDIUM,
                    description="Hips started level with or higher than shoulders, turning the pull into a stiff-leg deadlift.",
                    rep_number=rep_num,
                    frame=start_frame_num,
                    metric_value=start_hip["y"],
                    target_threshold=start_shoulder["y"],
                    phase="setup",
                )
            )

        # 2. Early Hip Rise ("Stripper Pull") Check during initial concentric pull
        mid_concentric_idx = start_idx + max(1, (lockout_idx - start_idx) // 3)
        mid_frame = frames[mid_concentric_idx]
        mid_hip = Kinematics.get_joint(mid_frame, "hip", dominant_side)
        mid_shoulder = Kinematics.get_joint(mid_frame, "shoulder", dominant_side)

        # Hip elevation vs shoulder elevation (remember smaller y is higher)
        hip_rise = start_hip["y"] - mid_hip["y"]
        shoulder_rise = start_shoulder["y"] - mid_shoulder["y"]

        metrics["initial_hip_rise"] = hip_rise
        metrics["initial_shoulder_rise"] = shoulder_rise

        # If hips shoot up significantly faster than shoulders while shoulders barely rise
        if hip_rise > 0.03 and shoulder_rise < (0.4 * hip_rise):
            faults.append(
                Fault(
                    code="EARLY_HIP_RISE",
                    name="Early Hip Rise (Stripper Pull)",
                    severity=FaultSeverity.HIGH,
                    description=f"Hips rose faster than chest off the floor (hip rise {hip_rise:.3f} vs chest {shoulder_rise:.3f}). Chest and hips must rise at the same rate.",
                    rep_number=rep_num,
                    frame=frames[mid_concentric_idx]["frame"],
                    metric_value=hip_rise / max(0.001, shoulder_rise),
                    target_threshold=2.0,
                    phase="concentric",
                )
            )

        # 3. Lockout & Hyperextension Check at Top
        lockout_frame = frames[lockout_idx]
        lock_hip = Kinematics.get_joint(lockout_frame, "hip", dominant_side)
        lock_knee = Kinematics.get_joint(lockout_frame, "knee", dominant_side)
        lock_shoulder = Kinematics.get_joint(lockout_frame, "shoulder", dominant_side)
        lock_ankle = Kinematics.get_joint(lockout_frame, "ankle", dominant_side)

        # Hip extension angle (shoulder-hip-knee)
        hip_extension_angle = Kinematics.calculate_angle_3p(lock_shoulder, lock_hip, lock_knee)
        # Check backward torso lean
        torso_angle = Kinematics.calculate_angle_with_vertical(lock_hip, lock_shoulder)
        # Check if shoulder is behind hip (backward hyperextension)
        leaning_backward = lock_shoulder["x"] < lock_hip["x"] if dominant_side == "left" else lock_shoulder["x"] > lock_hip["x"]

        metrics["lockout_hip_angle"] = hip_extension_angle
        metrics["lockout_torso_angle"] = torso_angle

        if hip_extension_angle < 165.0:
            faults.append(
                Fault(
                    code="INCOMPLETE_LOCKOUT",
                    name="Incomplete Hip Lockout",
                    severity=FaultSeverity.MEDIUM,
                    description=f"Hips did not reach full extension ({hip_extension_angle:.1f} deg, target >= 170 deg). Squeeze glutes at the top.",
                    rep_number=rep_num,
                    frame=lockout_frame_num,
                    metric_value=hip_extension_angle,
                    target_threshold=170.0,
                    phase="lockout",
                )
            )
        elif leaning_backward and torso_angle > self.max_hyperextension_angle:
            faults.append(
                Fault(
                    code="LUMBAR_HYPEREXTENSION",
                    name="Excessive Lumbar Hyperextension",
                    severity=FaultSeverity.HIGH,
                    description=f"Hyperextending lower back at lockout ({torso_angle:.1f} deg backward lean). Stand tall without arching spine back.",
                    rep_number=rep_num,
                    frame=lockout_frame_num,
                    metric_value=torso_angle,
                    target_threshold=self.max_hyperextension_angle,
                    phase="lockout",
                )
            )

        # 4. Bar Drift from Shins / Mid-foot
        # Wrist serves as barbell proxy if barbell itself is not segmented
        if f"{dominant_side}_wrist" in start_frame or "wrist" in start_frame:
            wrist = Kinematics.get_joint(lockout_frame, "wrist", dominant_side)
            bar_drift = abs(Kinematics.horizontal_drift(wrist, lock_ankle))
            metrics["bar_drift_at_lockout"] = bar_drift
            if bar_drift > self.max_bar_horizontal_drift:
                faults.append(
                    Fault(
                        code="BAR_DRIFT",
                        name="Bar Drift Away from Body",
                        severity=FaultSeverity.MEDIUM,
                        description=f"Bar drifted forward {bar_drift:.2f} relative to midfoot. Keep bar glued to shins and thighs.",
                        rep_number=rep_num,
                        frame=lockout_frame_num,
                        metric_value=bar_drift,
                        target_threshold=self.max_bar_horizontal_drift,
                        phase="concentric",
                    )
                )

        passed = not any(f.severity == FaultSeverity.HIGH for f in faults)

        return RepAnalysis(
            rep_number=rep_num,
            start_frame=start_frame_num,
            inflection_frame=lockout_frame_num,
            end_frame=end_frame_num,
            duration_seconds=duration_sec,
            eccentric_seconds=eccentric_sec,
            concentric_seconds=concentric_sec,
            passed=passed,
            faults=faults,
            metrics=metrics,
        )
