"""
TruckScenes dataset loader with temporal context, non-class padding
and simple data augmentation (Rauschen auf Box-Koordinaten).
"""

import numpy as np
from truckscenes import TruckScenes
from torch.utils.data import Dataset
from typing import Dict, List, Optional

class TruckScenesDataset(Dataset):
    """Temporal sensor fusion dataset for TruckScenes
    
    Features:
    - Loads full scene sequences
    - Maintains temporal ordering
    - Handles variable-length inputs via padding
    - Includes non-class (class 0) for empty slots
    - Adds Gaussian noise zu Box-Zentren (Datenaugmentation)
    """
    
    def __init__(
        self,
        dataroot: str,
        version: str,
        history_window: int = 5,
        max_boxes: int = 50,
        augment_noise_std: float = 0.1
    ):
        """
        Args:
            dataroot: Path to dataset root
            version: Dataset version (e.g., 'v1.0-mini')
            history_window: Number of historical frames to include
            max_boxes: Maximum boxes per frame after padding
            augment_noise_std: Standardabweichung des Gauss’schen Rauschens für [x,y,z]
        """
        self.ts = TruckScenes(version=version, dataroot=dataroot)
        self.history_window = history_window
        self.max_boxes = max_boxes
        self.augment_noise_std = augment_noise_std
        self.samples = self._get_chronological_samples()

    def _get_chronological_samples(self) -> List[str]:
        """Get ordered list of sample tokens maintaining scene sequence"""
        samples = []
        for scene in self.ts.scene:
            token = scene["first_sample_token"]
            while token:
                samples.append(token)
                token = self.ts.get("sample", token).get("next")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get sample with temporal context and augmentation
        
        Returns:
            Dictionary containing:
            - current: Current frame data (mit Random-Rauschen)
            - history: List of previous frames (ebenfalls augmentiert)
            - scene: Scene context identifier
            - padding_mask: Binary mask für real vs. padded entries
        """
        current_token = self.samples[idx]
        # Rohdaten laden
        current = self._load_sample(current_token)
        history_tokens = self._get_history_tokens(current_token)
        history = [self._load_sample(t) for t in history_tokens]

        # Datenaugmentation auf Box-Zentren
        current = self._augment_boxes(current)
        history = [self._augment_boxes(s) for s in history]

        return {
            "current": current,
            "history": history,
            "scene": self.ts.get("sample", current_token)["scene_token"],
            "padding_mask": self._create_padding_mask(current_token, history_tokens)
        }

    def _augment_boxes(self, sample: Optional[Dict]) -> Optional[Dict]:
        """Add Gaussian noise to the center coordinates [x,y,z] of each box"""
        if sample is None:
            return None
        boxes = sample["boxes"].copy()  # (N,7)
        if boxes.shape[0] > 0:
            noise = np.random.normal(
                loc=0.0,
                scale=self.augment_noise_std,
                size=(boxes.shape[0], 3)
            )
            boxes[:, :3] += noise  # Nur x,y,z stören
        return {"boxes": boxes}

    def _get_history_tokens(self, current_token: str) -> List[Optional[str]]:
        """Get list of up to `history_window` previous sample tokens"""
        tokens = []
        token = self.ts.get("sample", current_token).get("prev")
        for _ in range(self.history_window):
            if token:
                tokens.append(token)
                token = self.ts.get("sample", token).get("prev")
            else:
                tokens.append(None)
        return tokens

    def _create_padding_mask(self, current_token: str, history: List[Optional[str]]):
        """Create a mask (history+current) × max_boxes indicating real vs. padded entries"""
        mask = [0] * (len(history) + 1)
        for i, t in enumerate(history):
            mask[i] = 0 if t is None else 1
        mask[-1] = 1  # current immer valid
        return np.array(mask, dtype=np.uint8)

    def _load_sample(self, sample_token: Optional[str]) -> Optional[Dict]:
        """Load a single sample (keyframe) and convert its GT boxes to arrays
        
        Returns:
            None if sample_token is None (for padding), else dict {
                "boxes": np.ndarray of shape (N, 7) mit [x,y,z,w,l,h,yaw]
            }
        """
        if sample_token is None:
            return None

        sample = self.ts.get("sample", sample_token)
        arrs: List[np.ndarray] = []
        for ann in sample["anns"]:
            box = self.ts.get_box(ann)
            if hasattr(box, "to_array"):
                arr = box.to_array()
            else:
                center = box.center
                dims = box.wlh
                orient = box.orientation
                if hasattr(orient, "yaw_pitch_roll"):
                    yaw = orient.yaw_pitch_roll[0]
                elif hasattr(orient, "angle"):
                    yaw = orient.angle
                else:
                    yaw = float(orient)
                arr = np.array([
                    center[0], center[1], center[2],
                    dims[0], dims[1], dims[2],
                    yaw
                ], dtype=np.float32)
            arrs.append(arr)

        if arrs:
            boxes = np.vstack(arrs)
        else:
            boxes = np.zeros((0, 7), dtype=np.float32)

        return {"boxes": boxes}