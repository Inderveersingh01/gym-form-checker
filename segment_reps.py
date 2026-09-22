import json
import numpy as np
from scipy.signal import find_peaks
import sys

def segment_reps(keypoints_path):
    with open(keypoints_path) as f:
        data = json.load(f)

    frames = [f["frame"] for f in data]
    hip_y = np.array([f["left_hip"]["y"] for f in data])

    # Troughs = highest point of hip (standing) -> invert to find them as peaks
    # Peaks = lowest point of hip (bottom of squat, y is largest)
    peaks, _ = find_peaks(hip_y, prominence=0.05, distance=10)
    troughs, _ = find_peaks(-hip_y, prominence=0.05, distance=10)

    print(f"Total frames: {len(frames)}")
    print(f"Detected bottom-of-squat points (peaks in hip-y): {len(peaks)}")
    print(f"Frame indices of bottoms: {[frames[i] for i in peaks]}")
    print(f"Detected standing points (troughs in hip-y): {len(troughs)}")
    print(f"Frame indices of standing points: {[frames[i] for i in troughs]}")

    return peaks, troughs, hip_y, frames

if __name__ == "__main__":
    keypoints_path = sys.argv[1]
    segment_reps(keypoints_path)