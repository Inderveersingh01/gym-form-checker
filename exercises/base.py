import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any


class FaultSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Fault:
    code: str
    name: str
    severity: FaultSeverity
    description: str
    rep_number: int
    frame: int
    metric_value: float
    target_threshold: float
    phase: str  # e.g., "setup", "eccentric", "bottom", "concentric", "lockout"


@dataclass
class RepAnalysis:
    rep_number: int
    start_frame: int
    inflection_frame: int  # bottom of squat, chest-touch on bench, top of deadlift/pullup
    end_frame: int
    duration_seconds: float
    eccentric_seconds: float
    concentric_seconds: float
    passed: bool
    faults: List[Fault] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SetAnalysisResult:
    exercise: str
    total_reps: int
    successful_reps: int
    reps: List[RepAnalysis]
    overall_score: float  # 0.0 to 100.0
    detected_faults_summary: List[str]
    dominant_side: str
    summary_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exercise": self.exercise,
            "total_reps": self.total_reps,
            "successful_reps": self.successful_reps,
            "overall_score": round(self.overall_score, 1),
            "dominant_side": self.dominant_side,
            "detected_faults_summary": self.detected_faults_summary,
            "summary_metrics": self.summary_metrics,
            "reps": [
                {
                    "rep_number": r.rep_number,
                    "start_frame": r.start_frame,
                    "inflection_frame": r.inflection_frame,
                    "end_frame": r.end_frame,
                    "duration_seconds": round(r.duration_seconds, 2),
                    "eccentric_seconds": round(r.eccentric_seconds, 2),
                    "concentric_seconds": round(r.concentric_seconds, 2),
                    "passed": r.passed,
                    "metrics": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.metrics.items()},
                    "faults": [
                        {
                            "code": f.code,
                            "name": f.name,
                            "severity": f.severity.value,
                            "description": f.description,
                            "frame": f.frame,
                            "metric_value": round(f.metric_value, 2),
                            "target_threshold": round(f.target_threshold, 2),
                            "phase": f.phase,
                        }
                        for f in r.faults
                    ],
                }
                for r in self.reps
            ],
        }


class Kinematics:
    """
    Universal kinematic math utilities for 2D/3D human pose landmarks.
    Works directly with normalized coordinates (x in [0,1], y in [0,1]).
    Note: in screen space, y increases downwards (0 = top, 1 = bottom).
    """

    @staticmethod
    def calculate_angle_3p(
        a: Dict[str, float],
        b: Dict[str, float],
        c: Dict[str, float]
    ) -> float:
        """
        Calculates internal angle at vertex point b between segments ba and bc in degrees [0, 180].
        Example: a=hip, b=knee, c=ankle -> knee flexion angle.
        """
        ba_x = a["x"] - b["x"]
        ba_y = a["y"] - b["y"]
        bc_x = c["x"] - b["x"]
        bc_y = c["y"] - b["y"]

        dot = ba_x * bc_x + ba_y * bc_y
        mag_ba = math.hypot(ba_x, ba_y)
        mag_bc = math.hypot(bc_x, bc_y)

        if mag_ba * mag_bc == 0:
            return 0.0

        cosine = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
        angle_rad = math.acos(cosine)
        return math.degrees(angle_rad)

    @staticmethod
    def calculate_angle_with_vertical(
        p_bottom: Dict[str, float],
        p_top: Dict[str, float]
    ) -> float:
        """
        Calculates the acute angle of segment (p_bottom -> p_top) from vertical (gravity line) in degrees [0, 90].
        Example: p_bottom=hip, p_top=shoulder -> torso lean angle from vertical.
        """
        dx = abs(p_top["x"] - p_bottom["x"])
        # Screen coords: top point has smaller y
        dy = abs(p_bottom["y"] - p_top["y"])

        if dy == 0:
            return 90.0
        return math.degrees(math.atan2(dx, dy))

    @staticmethod
    def calculate_angle_with_horizontal(
        p1: Dict[str, float],
        p2: Dict[str, float]
    ) -> float:
        """
        Calculates angle of segment (p1 -> p2) with the horizontal in degrees [0, 90].
        Example: upper arm angle in bench press relative to horizontal floor.
        """
        dx = abs(p2["x"] - p1["x"])
        dy = abs(p2["y"] - p1["y"])
        if dx == 0:
            return 90.0
        return math.degrees(math.atan2(dy, dx))

    @staticmethod
    def euclidean_distance(
        p1: Dict[str, float],
        p2: Dict[str, float]
    ) -> float:
        """Calculates 2D Euclidean distance between two points."""
        return math.hypot(p1["x"] - p2["x"], p1["y"] - p2["y"])

    @staticmethod
    def horizontal_drift(
        p1: Dict[str, float],
        p2: Dict[str, float]
    ) -> float:
        """Calculates horizontal (x-axis) distance: p1['x'] - p2['x']."""
        return p1["x"] - p2["x"]

    @staticmethod
    def get_joint(frame: Dict[str, Any], joint_name: str, side: str = "left") -> Dict[str, float]:
        """
        Safely retrieves joint dict from frame dictionary.
        Supports automatic side prefixing ('hip' -> 'left_hip' or 'right_hip').
        """
        if joint_name in frame:
            return frame[joint_name]
        prefixed = f"{side}_{joint_name}"
        if prefixed in frame:
            return frame[prefixed]
        raise KeyError(f"Joint '{joint_name}' (or '{prefixed}') not found in frame keypoints.")


class BaseExerciseAnalyzer(ABC):
    """
    Abstract Base Class for all biomechanical movement rule engines.
    Every compound or isolation movement inherits from this class.
    """

    def __init__(self, exercise_name: str, fps: float = 30.0):
        self.exercise_name = exercise_name
        self.fps = fps

    @abstractmethod
    def get_primary_tracking_joint(self) -> Tuple[str, str]:
        """
        Returns (joint_name, axis) used for rep cycle segmentation.
        Example: ('hip', 'y') for squats, ('wrist', 'y') for bench press.
        """
        pass

    @abstractmethod
    def analyze_rep(
        self,
        frames: List[Dict[str, Any]],
        rep_info: Dict[str, int],
        dominant_side: str
    ) -> RepAnalysis:
        """
        Evaluates a single repetition cycle against movement-specific biomechanical rules.
        """
        pass

    def analyze_set(
        self,
        frames: List[Dict[str, Any]],
        reps_indices: List[Dict[str, int]],
        dominant_side: str = "left"
    ) -> SetAnalysisResult:
        """
        Evaluates the entire set of repetitions, aggregates faults, and computes overall score.
        """
        analyzed_reps: List[RepAnalysis] = []
        fault_counts: Dict[str, int] = {}
        total_faults = 0

        for rep_info in reps_indices:
            rep_analysis = self.analyze_rep(frames, rep_info, dominant_side)
            analyzed_reps.append(rep_analysis)

            for fault in rep_analysis.faults:
                fault_counts[fault.name] = fault_counts.get(fault.name, 0) + 1
                total_faults += 1

        total_reps = len(analyzed_reps)
        successful_reps = sum(1 for r in analyzed_reps if r.passed)

        # Compute overall set score (100 base, deduction for faults)
        if total_reps > 0:
            fault_penalty = sum(
                15 if f.severity == FaultSeverity.HIGH else (8 if f.severity == FaultSeverity.MEDIUM else 4)
                for r in analyzed_reps
                for f in r.faults
            )
            score = max(0.0, min(100.0, 100.0 - (fault_penalty / total_reps)))
        else:
            score = 0.0

        fault_summary = [
            f"{name} (detected in {count}/{total_reps} reps)"
            for name, count in fault_counts.items()
        ]

        return SetAnalysisResult(
            exercise=self.exercise_name,
            total_reps=total_reps,
            successful_reps=successful_reps,
            reps=analyzed_reps,
            overall_score=score,
            detected_faults_summary=fault_summary,
            dominant_side=dominant_side,
            summary_metrics=self._calculate_set_summary_metrics(analyzed_reps),
        )

    def _calculate_set_summary_metrics(self, reps: List[RepAnalysis]) -> Dict[str, Any]:
        """Calculates set-level aggregates like average tempo and consistency."""
        if not reps:
            return {}
        avg_duration = sum(r.duration_seconds for r in reps) / len(reps)
        avg_eccentric = sum(r.eccentric_seconds for r in reps) / len(reps)
        avg_concentric = sum(r.concentric_seconds for r in reps) / len(reps)
        return {
            "avg_rep_duration_sec": round(avg_duration, 2),
            "avg_eccentric_sec": round(avg_eccentric, 2),
            "avg_concentric_sec": round(avg_concentric, 2),
        }
