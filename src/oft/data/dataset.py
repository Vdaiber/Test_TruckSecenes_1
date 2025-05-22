# src/oft/data/dataset.py

import numpy as np
from torch.utils.data import Dataset
from typing import List, Optional, Dict

from oft.utils.common_utils import parse_scene_description
from oft.utils.config import load_config
from truckscenes import TruckScenes

class TruckScenesDataset(Dataset):
    """
    Lädt per TruckScenes-Devkit Samples und ergänzt:
      - alle Sample-Infos (prev/next, timestamp, scene_token)
      - alle sample_data-Records + calib + ego_pose pro Kanal
      - alle Annotation-Records + Instanz-, Attribut- und Visibility-Infos
      - current & history arrays (N×7: x,y,z,w,l,h,yaw)
      - velocities (N×3) berechnet aus erstem History-Frame
      - padding_mask (H+1,)
      - scene_meta (parsed description)
    """
    def __init__(
        self,
        dataroot: str,
        version: str,
        history_window: int = 5,
        max_boxes: int = 50,
        augment_noise_std: float = 0.0,
    ):
        self.ts = TruckScenes(version=version, dataroot=dataroot)
        self.history_window = history_window
        self.max_boxes = max_boxes
        self.augment_noise_std = augment_noise_std

        # chronologische Liste aller sample_tokens
        self.samples: List[str] = []
        for scene in self.ts.scene:
            tok = scene["first_sample_token"]
            while tok:
                self.samples.append(tok)
                tok = self.ts.get("sample", tok)["next"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        cfg = load_config()  # falls du noch pipeline-Parameter brauchst
        token = self.samples[idx]
        sample = self.ts.get("sample", token)

        # --- Sample-level meta ---
        prev_token   = sample["prev"]
        next_token   = sample["next"]
        scene_token  = sample["scene_token"]
        timestamp    = sample["timestamp"]
        ann_tokens   = sample["anns"]  # Liste der Annotation-Tokens

        # --- sample_data + calib + ego_pose pro Kanal ---
        sample_data  = {}
        calib_rec    = {}
        ego_pose_rec = {}
        for ch, sd_tok in sample["data"].items():
            sd = self.ts.get("sample_data", sd_tok)
            sample_data[ch]  = sd
            calib_rec[ch]    = self.ts.get("calibrated_sensor", sd["calibrated_sensor_token"])
            ego_pose_rec[ch] = self.ts.get("ego_pose",        sd["ego_pose_token"])

        # --- Annotation-Level Infos ---
        ann_rec     = {a: self.ts.get("sample_annotation", a) for a in ann_tokens}
        inst_rec    = {a: self.ts.get("instance",            ann_rec[a]["instance_token"])
                       for a in ann_tokens}
        attr_rec    = {a: [self.ts.get("attribute", at) for at in ann_rec[a]["attribute_tokens"]]
                       for a in ann_tokens}
        vis_rec     = {a: self.ts.get("visibility", ann_rec[a]["visibility_token"])
                       for a in ann_tokens if ann_rec[a]["visibility_token"]}

        # --- Szene-Meta (geparst) ---
        scene_rec   = self.ts.get("scene", scene_token)
        scene_meta  = parse_scene_description(scene_rec["description"])

        # --- 1) Current-Boxes extrahieren (N×7) ---
        arrs = []
        for a in ann_tokens:
            box = self.ts.get_box(a)
            cx, cy, cz = box.center
            w, l, h    = box.wlh
            orient     = box.orientation
            # yaw extrahieren
            if hasattr(orient, "yaw_pitch_roll"):
                yaw = orient.yaw_pitch_roll[0]
            elif hasattr(orient, "angle"):
                yaw = orient.angle
            else:
                yaw = float(orient)
            arrs.append(np.array([cx, cy, cz, w, l, h, yaw], dtype=np.float32))
        current = np.vstack(arrs) if arrs else np.zeros((0,7), dtype=np.float32)

        # --- 2) History-Frames laden ---
        history: List[Optional[np.ndarray]] = []
        prev_tok = sample["prev"]
        for _ in range(self.history_window):
            if prev_tok:
                s = self.ts.get("sample", prev_tok)
                arrs_h = []
                for a in s["anns"]:
                    box = self.ts.get_box(a)
                    cx, cy, cz = box.center
                    w, l, h    = box.wlh
                    orient     = box.orientation
                    if hasattr(orient, "yaw_pitch_roll"):
                        yaw = orient.yaw_pitch_roll[0]
                    elif hasattr(orient, "angle"):
                        yaw = orient.angle
                    else:
                        yaw = float(orient)
                    arrs_h.append(np.array([cx, cy, cz, w, l, h, yaw], dtype=np.float32))
                history.append(np.vstack(arrs_h) if arrs_h else np.zeros((0,7), dtype=np.float32))
                prev_tok = s["prev"]
            else:
                history.append(None)

        # --- 3) Padding-Maske (0=padd, 1=real) ---
        mask = [
            0 if (h is None or (isinstance(h, np.ndarray) and h.shape[0] == 0)) else 1
            for h in history
        ] + [1]  # current immer valid
        padding_mask = np.array(mask, dtype=np.uint8)  # Form: (history_window+1,)

        # --- 4) Gaussian-Noise auf current-Box-Zentren (optional) ---
        if self.augment_noise_std > 0 and current.shape[0] > 0:
            current[:, :3] += np.random.normal(
                loc=0.0,
                scale=self.augment_noise_std,
                size=(current.shape[0], 3)
            )

        # --- 5) Geschwindigkeiten aus erstem History-Frame (optional) ---
        if history and history[0] is not None and current.shape[0] == history[0].shape[0]:
            # einfache Differenz
            velocities = current[:, :3] - history[0][:, :3]
        else:
            velocities = None

        return {
            # sample-level
            "sample_token":     token,
            "prev":             prev_token,
            "next":             next_token,
            "scene_token":      scene_token,
            "timestamp":        timestamp,

            # sample_data + calib + ego_pose
            "sample_data":      sample_data,
            "calibrated_sensor":calib_rec,
            "ego_pose":         ego_pose_rec,

            # annotation-level
            "anns":             ann_tokens,
            "annotation":       ann_rec,
            "instance":         inst_rec,
            "attributes":       attr_rec,
            "visibility":       vis_rec,

            # parsed scene-meta
            "scene_meta":       scene_meta,

            # detected boxes
            "current":          current,     # (N,7)
            "history":          history,     # List[np.ndarray|None]
            "padding_mask":     padding_mask,# (H+1,)
            "velocities":       velocities,  # (N,3) or None
        }