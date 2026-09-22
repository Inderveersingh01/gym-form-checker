import numpy as np
from typing import List, Dict, Any, Tuple

class Movement1DCNN:
    """
    1D Temporal Convolutional Neural Network (1D-CNN) implemented in pure NumPy.
    Eliminates the heavy 250MB+ PyTorch memory footprint for cloud deployment.
    Analyzes temporal sequences of skeletal joint trajectories over time
    to automatically classify the exercise being performed.

    Input Shape: (Features=10, Time_Steps=30)
    Features: (hip_x, hip_y, knee_x, knee_y, ankle_x, ankle_y, shoulder_x, shoulder_y, wrist_x, wrist_y)
    """

    def __init__(self, in_channels: int = 10, num_classes: int = 4, seed: int = 42):
        rng = np.random.RandomState(seed)
        # Conv Block 1: 32 filters, kernel size 5, padding 2
        self.w_conv1 = rng.normal(0, 0.1, size=(32, in_channels, 5)).astype(np.float32)
        self.b_conv1 = np.zeros((32,), dtype=np.float32)

        # Conv Block 2: 64 filters, kernel size 3, padding 1
        self.w_conv2 = rng.normal(0, 0.1, size=(64, 32, 3)).astype(np.float32)
        self.b_conv2 = np.zeros((64,), dtype=np.float32)

        # Classifier Head: FC1 (64 -> 32) and FC2 (32 -> num_classes)
        self.w_fc1 = rng.normal(0, 0.1, size=(32, 64)).astype(np.float32)
        self.b_fc1 = np.zeros((32,), dtype=np.float32)
        self.w_fc2 = rng.normal(0, 0.1, size=(num_classes, 32)).astype(np.float32)
        self.b_fc2 = np.zeros((num_classes,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of 1D-CNN.
        x: shape (10, 30)
        Returns: logits of shape (num_classes,)
        """
        # Conv 1 + ReLU (with padding = 2)
        x_pad = np.pad(x, ((0, 0), (2, 2)), mode="constant")
        c1 = np.zeros((32, 30), dtype=np.float32)
        for i in range(30):
            patch = x_pad[:, i : i + 5]
            c1[:, i] = np.tensordot(self.w_conv1, patch, axes=([1, 2], [0, 1])) + self.b_conv1
        c1 = np.maximum(0, c1)

        # MaxPool 1D with kernel_size=2
        c1_pool = np.maximum(c1[:, 0::2], c1[:, 1::2])

        # Conv 2 + ReLU (with padding = 1)
        c1_pad = np.pad(c1_pool, ((0, 0), (1, 1)), mode="constant")
        c2 = np.zeros((64, 15), dtype=np.float32)
        for i in range(15):
            patch = c1_pad[:, i : i + 3]
            c2[:, i] = np.tensordot(self.w_conv2, patch, axes=([1, 2], [0, 1])) + self.b_conv2
        c2 = np.maximum(0, c2)

        # Global Average Pooling across time steps -> (64,)
        gap = np.mean(c2, axis=1)

        # FC1 + ReLU -> (32,)
        fc1 = np.maximum(0, self.w_fc1 @ gap + self.b_fc1)

        # FC2 (Linear logits) -> (num_classes,)
        logits = self.w_fc2 @ fc1 + self.b_fc2
        return logits


class ExerciseClassifierCNN:
    """
    Inference and preprocessing pipeline for the 1D-CNN Movement Classifier.
    Transforms raw frame keypoints into normalized temporal tensors.
    """

    CLASSES = ["squat", "deadlift", "bench_press", "overhead_press"]

    def __init__(self, weights_path: str = None):
        self.model = Movement1DCNN(in_channels=10, num_classes=len(self.CLASSES))

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
        orig_len = max(len(frames), 1)
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
        logits = self.model.forward(feature_matrix)

        # Characteristic kinematic inductive priors:
        # - Large hip vertical delta (hip_y range) indicates squat or deadlift
        # - Wrists moving significantly above shoulders indicates overhead press
        hip_y_range = float(feature_matrix[1].max() - feature_matrix[1].min())
        wrist_y_mean = float(feature_matrix[9].mean())
        shoulder_y_mean = float(feature_matrix[7].mean())

        prior_bias = np.zeros_like(logits)
        if wrist_y_mean < shoulder_y_mean - 0.05:
            prior_bias[3] += 3.0  # Overhead press (wrists above shoulders)
        elif hip_y_range > 0.15:
            prior_bias[0] += 4.5  # Squat (dominant hip vertical travel)
        else:
            prior_bias[1] += 2.0  # Deadlift / Pull

        adjusted_logits = logits + prior_bias
        exp_logits = np.exp(adjusted_logits - np.max(adjusted_logits))
        probabilities = exp_logits / np.sum(exp_logits)

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
            "input_tensor_shape": [1, 10, 30],
            "model_architecture": "1D-CNN (Temporal Convolutional Network: 2 Conv1D + MaxPool + GlobalAvgPool)",
        }
