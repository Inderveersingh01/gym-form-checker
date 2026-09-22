import numpy as np
from scipy.signal import find_peaks, savgol_filter
from typing import List, Dict, Any, Tuple


class RepDetector:
    """
    Universal repetition detector for cyclical resistance training movements.
    Uses signal processing (Savitzky-Golay smoothing and peak/trough analysis)
    to identify the exact lifecycle of each rep: start -> inflection -> end.
    """

    @staticmethod
    def smooth_signal(signal: np.ndarray, window_length: int = 7, polyorder: int = 2) -> np.ndarray:
        """Applies Savitzky-Golay filter to smooth joint coordinates and eliminate camera jitter."""
        if len(signal) <= window_length:
            return signal
        if window_length % 2 == 0:
            window_length += 1
        return savgol_filter(signal, window_length=window_length, polyorder=polyorder)

    @classmethod
    def segment_reps(
        cls,
        frames: List[Dict[str, Any]],
        joint_name: str = "hip",
        axis: str = "y",
        dominant_side: str = "left",
        invert: bool = False,
        prominence: float = 0.04,
        min_distance_frames: int = 15,
    ) -> List[Dict[str, int]]:
        """
        Segments repetition cycles from frame keypoints.

        For squat: hip-y reaches peak at the bottom (invert=False because y is down).
        For overhead press: wrist-y reaches peak at overhead (invert=True).
        """
        key = f"{dominant_side}_{joint_name}"
        raw_signal = []
        valid_indices = []

        for idx, f in enumerate(frames):
            if key in f and axis in f[key]:
                raw_signal.append(f[key][axis])
                valid_indices.append(idx)
            elif joint_name in f and axis in f[joint_name]:
                raw_signal.append(f[joint_name][axis])
                valid_indices.append(idx)

        if not raw_signal:
            return []

        signal = np.array(raw_signal)
        smoothed = cls.smooth_signal(signal)

        # If inverted, multiply by -1 to flip peaks/valleys
        analysis_signal = -smoothed if invert else smoothed

        # Inflections = deepest / peak position of each rep
        inflections, _ = find_peaks(analysis_signal, prominence=prominence, distance=min_distance_frames)
        # Reversals = top / standing positions
        valleys, _ = find_peaks(-analysis_signal, prominence=prominence * 0.7, distance=min_distance_frames // 2)

        reps: List[Dict[str, int]] = []

        for i, inflect_idx in enumerate(inflections):
            # Start of rep: closest valley before inflection
            prior_valleys = [v for v in valleys if v < inflect_idx]
            start_idx = prior_valleys[-1] if prior_valleys else max(0, inflect_idx - min_distance_frames)

            # End of rep: closest valley after inflection
            post_valleys = [v for v in valleys if v > inflect_idx]
            end_idx = post_valleys[0] if post_valleys else min(len(signal) - 1, inflect_idx + min_distance_frames)

            reps.append({
                "rep_number": i + 1,
                "start_idx": valid_indices[start_idx],
                "inflection_idx": valid_indices[inflect_idx],
                "end_idx": valid_indices[end_idx],
            })

        return reps
