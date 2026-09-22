import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from exercises.base import Kinematics, SetAnalysisResult


class VideoAnnotator:
    """
    OpenCV Video Renderer for Gym Form Feedback.
    Renders skeletal joints, angle arcs, real-time rep counter,
    and color-coded HUD status badges directly onto the video.
    """

    # Modern BGR color scheme
    COLOR_NEON_GREEN = (60, 235, 60)
    COLOR_CORAL_RED = (60, 60, 240)
    COLOR_AMBER = (30, 180, 255)
    COLOR_CYAN = (240, 210, 30)
    COLOR_WHITE = (255, 255, 255)
    COLOR_DARK_BG = (25, 25, 25)

    SKELETON_CONNECTIONS = [
        ("shoulder", "hip"),
        ("hip", "knee"),
        ("knee", "ankle"),
        ("shoulder", "elbow"),
        ("elbow", "wrist"),
    ]

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def _draw_hud_panel(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        alpha: float = 0.65
    ):
        """Draws a sleek semi-transparent dark HUD container."""
        sub_img = frame[y : y + h, x : x + w]
        black_rect = np.full_like(sub_img, self.COLOR_DARK_BG)
        res = cv2.addWeighted(sub_img, 1.0 - alpha, black_rect, alpha, 0)
        frame[y : y + h, x : x + w] = res
        cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 1)

    def render_annotated_video(
        self,
        video_path: str,
        output_path: str,
        frames_keypoints: List[Dict[str, Any]],
        set_result: SetAnalysisResult,
        dominant_side: str = "left",
        max_frames: Optional[int] = None
    ) -> str:
        """
        Renders annotated video with skeleton, angles, and live form feedback.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or self.fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))

        # Index keypoints by frame index
        keypoints_by_frame = {f["frame"]: f for f in frames_keypoints if "frame" in f}

        # Build active rep index mapping for fast frame lookup
        rep_map: Dict[int, Dict[str, Any]] = {}
        for rep in set_result.reps:
            for fr in range(rep.start_frame, rep.end_frame + 1):
                rep_map[fr] = {
                    "rep_number": rep.rep_number,
                    "phase": "DESCENT" if fr < rep.inflection_frame else "ASCENT",
                    "inflection": fr == rep.inflection_frame,
                    "depth_status": rep.metrics.get("depth_status", "PARALLEL"),
                    "passed": rep.passed,
                }

        current_rep_display = 0
        current_phase_display = "READY"
        active_badge = ("READY", self.COLOR_WHITE)

        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_idx >= max_frames:
                break

            kp = keypoints_by_frame.get(frame_idx)

            # Update rep state if available
            if frame_idx in rep_map:
                r_info = rep_map[frame_idx]
                current_rep_display = r_info["rep_number"]
                current_phase_display = r_info["phase"]

            # 1. Draw Skeleton Lines and Joint Markers
            if kp:
                coords = {}
                for joint in ["shoulder", "hip", "knee", "ankle", "elbow", "wrist"]:
                    j_dict = kp.get(f"{dominant_side}_{joint}", kp.get(joint))
                    if j_dict and "x" in j_dict and "y" in j_dict:
                        px = int(j_dict["x"] * width)
                        py = int(j_dict["y"] * height)
                        coords[joint] = (px, py)

                # Draw Bones
                for j1, j2 in self.SKELETON_CONNECTIONS:
                    if j1 in coords and j2 in coords:
                        cv2.line(frame, coords[j1], coords[j2], self.COLOR_CYAN, 3, cv2.LINE_AA)

                # Draw Joint Points
                for pt in coords.values():
                    cv2.circle(frame, pt, 6, self.COLOR_NEON_GREEN, -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, 8, self.COLOR_DARK_BG, 1, cv2.LINE_AA)

                # Compute Live Angles
                knee_angle = None
                torso_lean = None

                if "hip" in coords and "knee" in coords and "ankle" in coords:
                    hip_d = kp.get(f"{dominant_side}_hip", kp.get("hip"))
                    knee_d = kp.get(f"{dominant_side}_knee", kp.get("knee"))
                    ankle_d = kp.get(f"{dominant_side}_ankle", kp.get("ankle"))
                    knee_angle = Kinematics.calculate_angle_3p(hip_d, knee_d, ankle_d)

                    # Text label near knee
                    kx, ky = coords["knee"]
                    cv2.putText(
                        frame,
                        f"{knee_angle:.0f} deg",
                        (kx + 15, ky),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        self.COLOR_WHITE,
                        2,
                        cv2.LINE_AA,
                    )

                if "hip" in coords and "shoulder" in coords:
                    hip_d = kp.get(f"{dominant_side}_hip", kp.get("hip"))
                    sh_d = kp.get(f"{dominant_side}_shoulder", kp.get("shoulder"))
                    torso_lean = Kinematics.calculate_angle_with_vertical(hip_d, sh_d)

                # Determine HUD Badge
                if current_rep_display > 0:
                    if knee_angle and knee_angle <= 90.0:
                        active_badge = ("DEPTH: HIT", self.COLOR_NEON_GREEN)
                    elif torso_lean and torso_lean > 45.0:
                        active_badge = ("FORWARD LEAN", self.COLOR_CORAL_RED)
                    else:
                        active_badge = (f"{current_phase_display}", self.COLOR_AMBER)
                else:
                    active_badge = ("START POSITION", self.COLOR_WHITE)

            # 2. Draw Sleek Top-Left HUD Dashboard
            self._draw_hud_panel(frame, 20, 20, 320, 160, alpha=0.7)

            cv2.putText(
                frame,
                f"EXERCISE: {set_result.exercise.upper()}",
                (35, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                self.COLOR_CYAN,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"REP: {current_rep_display} / {set_result.total_reps}",
                (35, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                self.COLOR_WHITE,
                2,
                cv2.LINE_AA,
            )

            # Live Metrics
            k_text = f"Knee: {knee_angle:.0f} deg" if (kp and knee_angle) else "Knee: --"
            t_text = f"Torso: {torso_lean:.0f} deg" if (kp and torso_lean) else "Torso: --"
            cv2.putText(
                frame,
                f"{k_text} | {t_text}",
                (35, 118),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # Real-Time Form Status Badge
            badge_text, badge_color = active_badge
            cv2.rectangle(frame, (35, 132), (35 + len(badge_text) * 13 + 20, 162), badge_color, -1)
            cv2.putText(
                frame,
                badge_text,
                (45, 153),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        return output_path
