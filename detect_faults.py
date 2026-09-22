import json
import numpy as np
from scipy.signal import find_peaks
import sys

def detect_faults(keypoints_path, depth_threshold=0.02, back_angle_threshold=20):
    with open(keypoints_path) as f:
        data = json.load(f)

    frames = [f["frame"] for f in data]
    hip_y = np.array([f["left_hip"]["y"] for f in data])
    knee_y = np.array([f["left_knee"]["y"] for f in data])
    hip_x = np.array([f["left_hip"]["x"] for f in data])
    shoulder_x = np.array([f["left_shoulder"]["x"] for f in data])
    shoulder_y = np.array([f["left_shoulder"]["y"] for f in data])

    bottoms, _ = find_peaks(hip_y, prominence=0.05, distance=10)

    print(f"Analyzing {len(bottoms)} reps:\n")

    for i, idx in enumerate(bottoms):
        # Depth check: is hip_y >= knee_y (hip at or below knee) at the bottom?
        depth_diff = hip_y[idx] - knee_y[idx]
        depth_ok = depth_diff >= -depth_threshold  # allow small margin

        # Back angle: angle of hip-shoulder line from vertical
        dx = shoulder_x[idx] - hip_x[idx]
        dy = shoulder_y[idx] - hip_y[idx]
        angle_from_vertical = np.degrees(np.arctan2(abs(dx), abs(dy)))
        back_ok = angle_from_vertical <= back_angle_threshold

        print(f"Rep {i+1} (frame {frames[idx]}):")
        print(f"  Depth: hip_y={hip_y[idx]:.3f}, knee_y={knee_y[idx]:.3f}, diff={depth_diff:.3f} -> {'OK' if depth_ok else 'SHALLOW'}")
        print(f"  Back angle from vertical: {angle_from_vertical:.1f}° -> {'OK' if back_ok else 'ROUNDED/LEANING TOO MUCH'}")
        print()

if __name__ == "__main__":
    keypoints_path = sys.argv[1]
    detect_faults(keypoints_path)