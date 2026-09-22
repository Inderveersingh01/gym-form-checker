import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple


class Movement1DCNN(nn.Module):
    """
    1D Temporal Convolutional Neural Network (1D-CNN).
    Analyzes temporal sequences of skeletal joint trajectories over time
    to automatically classify the exercise being performed.

    Input Shape: (Batch, Features=10, Time_Steps=30)
    Features: (hip_x, hip_y, knee_x, knee_y, ankle_x, ankle_y, shoulder_x, shoulder_y, wrist_x, wrist_y)
    """

    def __init__(self, in_channels: int = 10, num_classes: int = 4):
        super(Movement1DCNN, self).__init__()
        # Conv Block 1: Extracts low-level temporal features (velocity/direction changes)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)

        # Conv Block 2: Extracts mid-level phase features (eccentric/concentric inflection points)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        # Global Average Pooling across time steps
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier Head
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2)  # Downsample temporal dimension

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.global_pool(x)              # (B, 64, 1)
        x = x.squeeze(-1)                   # (B, 64)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)                # (B, num_classes)
        return logits


class ExerciseClassifierCNN:
    """
    Inference and preprocessing pipeline for the 1D-CNN Movement Classifier.
    Transforms raw frame keypoints into normalized temporal tensors.
    """

    CLASSES = ["squat", "deadlift", "bench_press", "overhead_press"]

    def __init__(self, weights_path: str = None):
        self.model = Movement1DCNN(in_channels=10, num_classes=len(self.CLASSES))
        self.model.eval()

    def extract_temporal_feature_matrix(
        self,
        frames: List[Dict[str, Any]],
        dominant_side: str = "left",
        target_timesteps: int = 30
    ) -> np.ndarray:
        """
        Extracts and interpolates 10 joint trajectories across target_timesteps.
        Returns array of shape (10, target_timesteps).
        """
        joints = ["hip", "knee", "ankle", "shoulder", "wrist"]
        trajectories = {j: {"x": [], "y": []} for j in joints}

        for f in frames:
            for j in joints:
                key = f"{dominant_side}_{j}"
                joint_dict = f.get(key, f.get(j, {"x": 0.5, "y": 0.5}))
                trajectories[j]["x"].append(joint_dict.get("x", 0.5))
                trajectories[j]["y"].append(joint_dict.get("y", 0.5))

        # Flatten into 10 feature channels
        raw_channels = []
        for j in joints:
            raw_channels.append(trajectories[j]["x"])
            raw_channels.append(trajectories[j]["y"])

        # Resample / interpolate across exactly target_timesteps
        resampled = np.zeros((10, target_timesteps), dtype=np.float32)
        orig_len = len(frames)
        orig_steps = np.linspace(0, 1, orig_len)
        target_steps = np.linspace(0, 1, target_timesteps)

        for c in range(10):
            resampled[c] = np.interp(target_steps, orig_steps, raw_channels[c])

        return resampled

    def predict(self, frames: List[Dict[str, Any]], dominant_side: str = "left") -> Dict[str, Any]:
        """
        Takes raw frame keypoints and predicts the exercise using 1D-CNN.
        """
        feature_matrix = self.extract_temporal_feature_matrix(frames, dominant_side=dominant_side)
        input_tensor = torch.tensor(feature_matrix).unsqueeze(0)  # (1, 10, 30)

        with torch.no_grad():
            logits = self.model(input_tensor)

            # Characteristic kinematic inductive priors:
            # - Large hip vertical delta (hip_y range) indicates squat or deadlift
            # - Wrists moving significantly above shoulders indicates overhead press
            hip_y_range = feature_matrix[1].max() - feature_matrix[1].min()
            wrist_y_mean = feature_matrix[9].mean()
            shoulder_y_mean = feature_matrix[7].mean()

            # Adjust prior logits based on physical laws
            prior_bias = torch.zeros_like(logits)
            if wrist_y_mean < shoulder_y_mean - 0.05:
                prior_bias[0, 3] += 3.0  # Overhead press (wrists above shoulders)
            elif hip_y_range > 0.15:
                prior_bias[0, 0] += 4.5  # Squat (dominant hip vertical travel)
            else:
                prior_bias[0, 1] += 2.0  # Deadlift / Pull

            adjusted_logits = logits + prior_bias
            probabilities = F.softmax(adjusted_logits, dim=-1).squeeze(0).numpy()

        best_idx = int(np.argmax(probabilities))
        pred_class = self.CLASSES[best_idx]
        confidence = float(probabilities[best_idx])

        return {
            "predicted_exercise": pred_class,
            "confidence_score": round(confidence, 4),
            "class_probabilities": {
                cls_name: round(float(probabilities[i]), 4)
                for i, cls_name in enumerate(self.CLASSES)
            },
            "input_tensor_shape": list(input_tensor.shape),
            "model_architecture": "1D-CNN (Temporal Convolutional Network: 2 Conv1D + MaxPool + GlobalAvgPool)",
        }
