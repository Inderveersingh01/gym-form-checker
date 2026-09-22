from typing import Dict, Type
from exercises.base import BaseExerciseAnalyzer, Kinematics, RepAnalysis, Fault, FaultSeverity, SetAnalysisResult
from exercises.squat import SquatAnalyzer
from exercises.deadlift import DeadliftAnalyzer
from exercises.bench_press import BenchPressAnalyzer
from exercises.overhead_press import OverheadPressAnalyzer

EXERCISE_REGISTRY: Dict[str, Type[BaseExerciseAnalyzer]] = {
    "squat": SquatAnalyzer,
    "back_squat": SquatAnalyzer,
    "front_squat": SquatAnalyzer,
    "goblet_squat": SquatAnalyzer,
    "deadlift": DeadliftAnalyzer,
    "rdl": DeadliftAnalyzer,
    "romanian_deadlift": DeadliftAnalyzer,
    "bench_press": BenchPressAnalyzer,
    "pushup": BenchPressAnalyzer,
    "push_up": BenchPressAnalyzer,
    "overhead_press": OverheadPressAnalyzer,
    "ohp": OverheadPressAnalyzer,
    "military_press": OverheadPressAnalyzer,
}

def get_analyzer(exercise_name: str, fps: float = 30.0, **kwargs) -> BaseExerciseAnalyzer:
    """
    Factory function to retrieve the appropriate biomechanical analyzer for any supported movement.
    """
    key = exercise_name.lower().replace(" ", "_").replace("-", "_")
    if key not in EXERCISE_REGISTRY:
        supported = ", ".join(sorted(set(EXERCISE_REGISTRY.keys())))
        raise ValueError(f"Unsupported exercise '{exercise_name}'. Supported: {supported}")
    
    cls = EXERCISE_REGISTRY[key]
    return cls(fps=fps, **kwargs)

__all__ = [
    "BaseExerciseAnalyzer",
    "Kinematics",
    "RepAnalysis",
    "Fault",
    "FaultSeverity",
    "SetAnalysisResult",
    "SquatAnalyzer",
    "DeadliftAnalyzer",
    "BenchPressAnalyzer",
    "OverheadPressAnalyzer",
    "get_analyzer",
    "EXERCISE_REGISTRY",
]
