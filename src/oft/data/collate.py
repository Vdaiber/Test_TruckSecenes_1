# src/oft/data/collate.py

import torch
import numpy as np
from typing import Dict, List, Any

from oft.utils.config import load_config

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pads to pipeline.yaml‘s gt_max_boxes (if not null) or to the batch-max,
    und returned **alle** Felder aus dem Dataset plus padded tensors:
      - current (B, M, 7)
      - history (B, H, M, 7)
      - padding_mask (B, H+1, M)
      - dazu: sample_token, prev, next, scene_token, timestamp,
        sample_data, calibrated_sensor, ego_pose, anns, annotation,
        instance, attributes, visibility, velocities, scene_meta
    """
    cfg       = load_config()
    fixed_max = cfg.get("gt_max_boxes", None)

    # 1) Hole alle Roh-Felder
    sample_tokens     = [b["sample_token"]     for b in batch]
    prevs             = [b["prev"]             for b in batch]
    nexts             = [b["next"]             for b in batch]
    scene_tokens      = [b["scene_token"]      for b in batch]
    timestamps        = [b["timestamp"]        for b in batch]
    sample_datas      = [b["data"]             for b in batch]  # für Visualizer
    calibrated_sensors= [b["calibrated_sensor"]for b in batch]
    ego_poses         = [b["ego_pose"]         for b in batch]
    anns_list         = [b["anns"]             for b in batch]
    annotations       = [b["annotation"]       for b in batch]
    instances         = [b["instance"]         for b in batch]
    attributes_list   = [b["attributes"]       for b in batch]
    visibilities      = [b["visibility"]       for b in batch]
    scene_metas       = [b["scene_meta"]       for b in batch]
    velocities        = [b.get("velocities")   for b in batch]  # kann None sein

    currents   = [b["current"]      for b in batch]  # np.ndarray (Ni,7)
    histories  = [b["history"]      for b in batch]  # List[List[np.ndarray|None]]
    raw_masks  = [b["padding_mask"] for b in batch]  # np.ndarray (H+1,)

    B = len(batch)

    # 2) Bestimme M = fixed_max oder dynamisch
    if fixed_max is not None:
        M = fixed_max
    else:
        M = max((arr.shape[0] for arr in currents), default=0)

    # 3) Pad current → (B, M, 7)
    frames = torch.zeros((B, M, 7), dtype=torch.float32)
    for i, arr in enumerate(currents):
        n = min(arr.shape[0], M)
        if n:
            frames[i, :n] = torch.from_numpy(arr[:n])

    # 4) Pad history → (B, H, M, 7)
    H = max(len(h) for h in histories)
    hist = torch.zeros((B, H, M, 7), dtype=torch.float32)
    for i, hlist in enumerate(histories):
        for t, arr in enumerate(hlist):
            if arr is not None and arr.shape[0] > 0:
                n = min(arr.shape[0], M)
                hist[i, t, :n] = torch.from_numpy(arr[:n])

    # 5) Baue 3D-Padding-Maske (B, H+1, M)
    masks_1d = torch.stack(
        [torch.from_numpy(m.astype(np.uint8)) for m in raw_masks],
        dim=0
    )  # (B, H+1)
    mask_3d = masks_1d.unsqueeze(-1).expand(-1, -1, M)

    return {
        # gebatchete Tensors
        "current":       frames,         # torch.Tensor (B, M, 7)
        "history":       hist,           # torch.Tensor (B, H, M, 7)
        "padding_mask":  mask_3d,        # torch.Tensor (B, H+1, M)

        # alle übrigen Metadaten als Listen
        "sample_token":      sample_tokens,
        "prev":              prevs,
        "next":              nexts,
        "scene_token":       scene_tokens,
        "timestamp":         timestamps,
        "sample_data":       sample_datas,
        "calibrated_sensor": calibrated_sensors,
        "ego_pose":          ego_poses,
        "anns":              anns_list,
        "annotation":        annotations,
        "instance":          instances,
        "attributes":        attributes_list,
        "visibility":        visibilities,
        "scene_meta":        scene_metas,
        "velocities":        velocities,
    }