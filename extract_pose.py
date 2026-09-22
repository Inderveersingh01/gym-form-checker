import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import sys

def extract_keypoints(video_path, model_path="pose_landmarker.task", sample_every_n_frames=2):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    results_per_frame = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every_n_frames == 0:
            h, w = frame.shape[:2]
            if max(h, w) > 640:
                scale = 640.0 / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / fps) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                keypoints = {
                    "frame": frame_idx,
                    "left_hip": {"x": lm[23].x, "y": lm[23].y, "vis": getattr(lm[23], "visibility", None)},
                    "right_hip": {"x": lm[24].x, "y": lm[24].y, "vis": getattr(lm[24], "visibility", None)},
                    "left_knee": {"x": lm[25].x, "y": lm[25].y, "vis": getattr(lm[25], "visibility", None)},
                    "right_knee": {"x": lm[26].x, "y": lm[26].y, "vis": getattr(lm[26], "visibility", None)},
                    "left_ankle": {"x": lm[27].x, "y": lm[27].y, "vis": getattr(lm[27], "visibility", None)},
                    "right_ankle": {"x": lm[28].x, "y": lm[28].y, "vis": getattr(lm[28], "visibility", None)},
                    "left_shoulder": {"x": lm[11].x, "y": lm[11].y, "vis": getattr(lm[11], "visibility", None)},
                    "right_shoulder": {"x": lm[12].x, "y": lm[12].y, "vis": getattr(lm[12], "visibility", None)},
                    "left_elbow": {"x": lm[13].x, "y": lm[13].y, "vis": getattr(lm[13], "visibility", None)},
                    "right_elbow": {"x": lm[14].x, "y": lm[14].y, "vis": getattr(lm[14], "visibility", None)},
                    "left_wrist": {"x": lm[15].x, "y": lm[15].y, "vis": getattr(lm[15], "visibility", None)},
                    "right_wrist": {"x": lm[16].x, "y": lm[16].y, "vis": getattr(lm[16], "visibility", None)},
                }
                results_per_frame.append(keypoints)

        frame_idx += 1

    cap.release()
    landmarker.close()
    return results_per_frame

if __name__ == "__main__":
    video_path = sys.argv[1]
    data = extract_keypoints(video_path)
    out_path = video_path.rsplit(".", 1)[0] + "_keypoints.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Extracted {len(data)} frames of keypoints -> {out_path}")