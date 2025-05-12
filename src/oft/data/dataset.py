import numpy as np
from torch.utils.data import Dataset
from typing import List, Optional, Dict
from oft.utils.common_utils import parse_scene_description


from truckscenes import TruckScenes

class TruckScenesDataset(Dataset):
    """
    Lädt per TruckScenes-Tutorial Samples und ergänzt temporal
    Context (History) sowie Non-Class-Padding und optionale Gaussian-Noise.
    """
    def __init__(
        self,
        dataroot: str,
        version: str,
        history_window: int = 5,
        max_boxes: int = 50,
        augment_noise_std: float = 0.0
    ):
        """
        Args:
            dataroot: Pfad zum Ordner, der 'v1.0-mini' enthält
            version: Dataset-Version (z.B. 'v1.0-mini')
            history_window: Anzahl vorheriger Frames
            max_boxes: fixe Obergrenze für Boxen pro Frame
            augment_noise_std: SD für Gaussian Noise auf Box-Zentren
        """
        self.ts = TruckScenes(version=version, dataroot=dataroot)
        self.history_window = history_window
        self.max_boxes = max_boxes
        self.augment_noise_std = augment_noise_std

        # chronologische Liste aller Sample-Tokens
        self.samples: List[str] = []
        for scene in self.ts.scene:
            tok = scene["first_sample_token"]
            while tok:
                self.samples.append(tok)
                tok = self.ts.get("sample", tok)["next"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        token = self.samples[idx]
        sample = self.ts.get("sample", token)

        # --- Szene-Metadaten (parsed description) ---
        scene_token = sample["scene_token"]
        scene_rec   = self.ts.get("scene", scene_token)
        scene_meta  = parse_scene_description(scene_rec["description"])
        # --- 1) Current-Boxes extrahieren (N×7) ---
        arrs = []
        for ann_tk in sample["anns"]:
            box = self.ts.get_box(ann_tk)
            cx, cy, cz = box.center
            w, l, h = box.wlh
            orient = box.orientation
            # yaw extrahieren
            if hasattr(orient, "yaw_pitch_roll"):
                yaw = orient.yaw_pitch_roll[0]
            elif hasattr(orient, "angle"):
                yaw = orient.angle
            else:
                yaw = float(orient)
            arrs.append(np.array([cx, cy, cz, w, l, h, yaw], dtype=np.float32))
        current = np.vstack(arrs) if arrs else np.zeros((0, 7), dtype=np.float32)

        # --- 2) History-Frames laden ---
        history: List[Optional[np.ndarray]] = []
        prev = sample["prev"]
        for _ in range(self.history_window):
            if prev:
                s = self.ts.get("sample", prev)
                arrs_h = []
                for ann_tk in s["anns"]:
                    box = self.ts.get_box(ann_tk)
                    cx, cy, cz = box.center
                    w, l, h = box.wlh
                    orient = box.orientation
                    if hasattr(orient, "yaw_pitch_roll"):
                        yaw = orient.yaw_pitch_roll[0]
                    elif hasattr(orient, "angle"):
                        yaw = orient.angle
                    else:
                        yaw = float(orient)
                    arrs_h.append(np.array([cx, cy, cz, w, l, h, yaw], dtype=np.float32))
                history.append(np.vstack(arrs_h) if arrs_h else np.zeros((0,7),dtype=np.float32))
                prev = s["prev"]
            else:
                history.append(None)

        # --- 3) Padding-Maske (0=padd, 1=real) ---
        mask = [
            0 if (h is None or (isinstance(h, np.ndarray) and h.shape[0] == 0)) else 1
            for h in history
        ] + [1]  # current immer valid
        padding_mask = np.array(mask, dtype=np.uint8)  # Form: (history_window+1,)

        # --- 4) Optional: Gaussian-Noise auf current-Box-Zentren ---
        if self.augment_noise_std > 0 and current.shape[0] > 0:
            current[:, :3] += np.random.normal(
                loc=0.0,
                scale=self.augment_noise_std,
                size=(current.shape[0], 3)
            )

        return {
            "current": current,           # np.ndarray (N,7)
            "history": history,           # List[np.ndarray|None]
            "padding_mask": padding_mask, # np.ndarray (history_window+1,)
            "scene": sample["scene_token"],
            "scene_meta": scene_meta,     # parsed scene description
            "timestamp": sample["timestamp"],
            "data": sample["data"],       # für spätere Visualisierung
        }