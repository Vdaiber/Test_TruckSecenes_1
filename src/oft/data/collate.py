# oft/data/collate.py

import torch
import numpy as np
from typing import Dict, List

from oft.utils.config import load_config

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Pads to pipeline.yaml‘s gt_max_boxes (if not null) or to the batch-max,
    and returns current/history/mask + scene tokens + scene_meta dicts.
    """
    cfg       = load_config()
    fixed_max = cfg.get("gt_max_boxes", None)

    currents   = [b["current"] for b in batch]      # np.ndarray (Ni,7)
    histories  = [b["history"] for b in batch]      # List[List[np.ndarray|None]]
    raw_masks  = [b["padding_mask"] for b in batch] # np.ndarray (H+1,)
    scenes     = [b["scene"] for b in batch]        # str tokens
    metas      = [b["scene_meta"] for b in batch]   # dict per sample

    B = len(currents)

    # 1) decide M = fixed_max or dynamic max over currents
    if fixed_max is not None:
        M = fixed_max
    else:
        M = max((arr.shape[0] for arr in currents), default=0)

    # 2) Pad current to (B, M, 7)
    frames = torch.zeros((B, M, 7), dtype=torch.float32)
    for i, arr in enumerate(currents):
        n = min(arr.shape[0], M)
        if n:
            frames[i, :n] = torch.from_numpy(arr[:n])

    # 3) Pad history to (B, H, M, 7)
    H = max(len(h) for h in histories)
    hist = torch.zeros((B, H, M, 7), dtype=torch.float32)
    for i, hlist in enumerate(histories):
        for t, arr in enumerate(hlist):
            if arr is not None and arr.shape[0] > 0:
                n = min(arr.shape[0], M)
                hist[i, t, :n] = torch.from_numpy(arr[:n])

    # 4) Build 3D padding mask (B, H+1, M)
    masks_1d = torch.stack(
        [torch.from_numpy(m.astype(np.uint8)) for m in raw_masks],
        dim=0
    )  # (B, H+1)
    mask_3d = masks_1d.unsqueeze(-1).expand(-1, -1, M)

    return {
        "current":      frames,        # torch.Tensor (B, M, 7)
        "history":      hist,          # torch.Tensor (B, H, M, 7)
        "padding_mask": mask_3d,       # torch.Tensor (B, H+1, M)
        "scene":        scenes,        # List[str]
        "scene_meta":   metas,         # List[Dict]
    }