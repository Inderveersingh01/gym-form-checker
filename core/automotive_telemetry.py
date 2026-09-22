import math
from typing import Any, Dict, List, Optional, Tuple


class KalmanFilter1D:
    """
    Classic State-Space Kalman Filter (Estimates Position and Velocity).
    Same algorithm used in Autonomous Vehicles (ADAS) to filter noisy sensor
    measurements (cameras/radar) and predict trajectories through occlusion.
    """

    def __init__(self, process_noise: float = 1e-4, measurement_noise: float = 1e-2):
        self.x = 0.0     # Estimated position
        self.v = 0.0     # Estimated velocity
        self.p_xx = 1.0  # Estimation error covariance
        self.p_vv = 1.0
        self.p_xv = 0.0
        self.q = process_noise
        self.r = measurement_noise
        self.initialized = False

    def update(self, measurement: float, dt: float = 1.0 / 30.0) -> Tuple[float, float]:
        """
        Runs the standard Predict -> Measurement Update Kalman cycle.
        Returns: (filtered_position, filtered_velocity)
        """
        if not self.initialized:
            self.x = measurement
            self.v = 0.0
            self.initialized = True
            return self.x, self.v

        # 1. State Prediction (Physics: x = x + v*dt)
        self.x = self.x + self.v * dt
        self.p_xx += dt * (2.0 * self.p_xv + dt * self.p_vv) + self.q
        self.p_xv += dt * self.p_vv
        self.p_vv += self.q

        # 2. Measurement Update (Innovation)
        residual = measurement - self.x
        s = self.p_xx + self.r
        k_x = self.p_xx / s
        k_v = self.p_xv / s

        # 3. State Correction
        self.x += k_x * residual
        self.v += k_v * residual

        # 4. Covariance Correction
        self.p_xx *= (1.0 - k_x)
        self.p_xv *= (1.0 - k_x)
        self.p_vv -= k_v * self.p_xv

        return self.x, self.v


class LaneDepartureWarning:
    """
    Biomechanical ADAS: Lane Departure Warning (LDW).
    In autonomous driving, LDW flags when tires drift out of highway lane markings.
    In lifting, optimal physics requires the bar/load to stay inside a vertical 'lane'
    centered over the lifter's midfoot (center of balance).
    """

    def __init__(self, lane_tolerance_pct: float = 0.05):
        # 0.05 = 5% of screen width tolerance from midfoot center
        self.lane_tolerance = lane_tolerance_pct

    def check_drift(self, bar_x: float, midfoot_x: float) -> Tuple[bool, float, Optional[str]]:
        """
        Calculates drift from midfoot lane center.
        Returns: (is_departure, drift_amount, warning_label)
        """
        drift = bar_x - midfoot_x
        if drift > self.lane_tolerance:
            return True, drift, "LANE DEPARTURE: BAR DRIFTING FORWARD"
        elif drift < -self.lane_tolerance:
            return True, drift, "LANE DEPARTURE: BAR DRIFTING BACKWARD"
        return False, drift, None


class TelematicsEngine:
    """
    Vehicle-style Telemetry (Velocity-Based Training & Fatigue Analytics).
    Tracks velocity degradation across repetitions to mathematically detect
    neuromuscular fatigue before mechanical breakdown occurs.
    """

    @staticmethod
    def analyze_velocity_loss(
        rep_velocities: List[float],
        fatigue_threshold_pct: float = 25.0
    ) -> Dict[str, Any]:
        """
        Computes velocity loss from the fastest initial rep to the final rep.
        Velocity Loss (%) = ((V_fastest - V_final) / V_fastest) * 100
        """
        if not rep_velocities:
            return {"fatigue_detected": False, "velocity_loss_pct": 0.0}

        fastest_vel = max(rep_velocities)
        final_vel = rep_velocities[-1]

        if fastest_vel <= 0:
            loss_pct = 0.0
        else:
            loss_pct = ((fastest_vel - final_vel) / fastest_vel) * 100.0
            loss_pct = max(0.0, loss_pct)

        fatigue_alert = loss_pct >= fatigue_threshold_pct

        return {
            "fastest_rep_concentric_vel": round(fastest_vel, 3),
            "final_rep_concentric_vel": round(final_vel, 3),
            "velocity_loss_pct": round(loss_pct, 1),
            "fatigue_threshold_pct": fatigue_threshold_pct,
            "fatigue_detected": fatigue_alert,
            "telematics_status": "HIGH FATIGUE / RPE 9.5+" if fatigue_alert else "STABLE MOTOR UNIT OUTPUT",
        }


class AutomotiveKinematicsTelemetry:
    """
    Unified High-Level Interface: Combines Kalman Filtering,
    Lane Departure Warning (LDW), and Velocity Telematics.
    """

    def __init__(self, fps: float = 30.0, lane_tolerance_pct: float = 0.05):
        self.fps = fps
        self.dt = 1.0 / fps
        self.ldw = LaneDepartureWarning(lane_tolerance_pct=lane_tolerance_pct)
        self.kalman_x = KalmanFilter1D()
        self.kalman_y = KalmanFilter1D()

    def process_telemetry(
        self,
        frames: List[Dict[str, Any]],
        reps: List[Dict[str, int]],
        dominant_side: str = "left"
    ) -> Dict[str, Any]:
        """
        Runs automotive telematics across all video frames and segmented reps.
        """
        hip_key = f"{dominant_side}_hip"
        ankle_key = f"{dominant_side}_ankle"
        shoulder_key = f"{dominant_side}_shoulder"

        lane_departures = []
        filtered_trajectory = []

        # 1. Kalman Filter & Lane Departure across all frames
        for f in frames:
            frame_num = f.get("frame", 0)
            hip = f.get(hip_key, f.get("hip"))
            ankle = f.get(ankle_key, f.get("ankle"))
            shoulder = f.get(shoulder_key, f.get("shoulder"))

            if not hip or not ankle:
                continue

            # Load tracking: shoulder acts as barbell proxy in squats
            load_point = shoulder if shoulder else hip
            midfoot_x = ankle["x"]

            # Apply Kalman Filter to load position (x and y)
            smooth_x, vel_x = self.kalman_x.update(load_point["x"], dt=self.dt)
            smooth_y, vel_y = self.kalman_y.update(load_point["y"], dt=self.dt)

            filtered_trajectory.append({
                "frame": frame_num,
                "raw_x": round(load_point["x"], 4),
                "smooth_x": round(smooth_x, 4),
                "raw_y": round(load_point["y"], 4),
                "smooth_y": round(smooth_y, 4),
                "velocity_y": round(vel_y, 4),
            })

            # Check ADAS Lane Departure (Bar vs Midfoot)
            is_dep, drift, warning = self.ldw.check_drift(smooth_x, midfoot_x)
            if is_dep:
                lane_departures.append({
                    "frame": frame_num,
                    "drift_pct": round(drift * 100.0, 1),
                    "warning": warning,
                })

        # 2. Compute Velocity Loss across concentric phases of segmented reps
        rep_concentric_velocities = []
        for r in reps:
            start_idx = r["inflection_idx"]
            end_idx = r["end_idx"]
            concentric_frames = max(1, end_idx - start_idx)
            concentric_duration = concentric_frames / self.fps

            # Vertical displacement of hip during concentric drive
            start_y = frames[start_idx].get(hip_key, {}).get("y", 0.0)
            end_y = frames[end_idx].get(hip_key, {}).get("y", 0.0)
            vertical_dist = abs(end_y - start_y)

            # Velocity = distance / time (normalized units per second)
            mean_concentric_vel = vertical_dist / concentric_duration if concentric_duration > 0 else 0.0
            rep_concentric_velocities.append(mean_concentric_vel)

        vbt_telematics = TelematicsEngine.analyze_velocity_loss(rep_concentric_velocities)

        return {
            "total_frames_analyzed": len(filtered_trajectory),
            "lane_departure_events_count": len(lane_departures),
            "lane_departures_sample": lane_departures[:3],  # First few events
            "concentric_velocities_per_rep": [round(v, 3) for v in rep_concentric_velocities],
            "velocity_telematics": vbt_telematics,
        }
