"""
Custom collate function for temporal sensor fusion.

Pads:
- Variable-length current-frame detections
- Variable-length history sequences
- Marks padded entries with non-class ID in feature dimension
"""

import torch
import numpy as np
from typing import Dict, List, Optional

NON_CLASS_FEATURE_IDX = 0  # index of the class field in features

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Batch a list of samples into tensors ready for the Transformer.

    Returns:
      - current: (B, max_boxes, 7)
      - history: (B, history_window, max_boxes, 7)
      - padding_mask: (B, history_window+1, max_boxes)
      - scene: List[str]
    """
    currents = [b["current"] for b in batch]
    histories = [b["history"] for b in batch]
    raw_masks = [b["padding_mask"] for b in batch]  # each shape: (history_window+1,)
    scenes = [b["scene"] for b in batch]

    # Pad frames and histories
    frames = _pad_frames(currents)         # shape: (B, max_boxes, 7)
    hist = _pad_histories(histories)      # shape: (B, H, max_boxes, 7)

    # Build 3D padding mask from 1D masks
    B, max_boxes, _ = frames.shape
    # Convert numpy masks to torch (B, H+1)
    masks_1d = torch.stack([torch.from_numpy(m.astype(np.uint8)) for m in raw_masks], dim=0)
    # Expand to (B, H+1, max_boxes)
    mask_3d = masks_1d.unsqueeze(-1).expand(-1, -1, max_boxes)

    return {
        "current": frames,
        "history": hist,
        "padding_mask": mask_3d,
        "scene": scenes
    }

def _pad_frames(samples: List[Optional[Dict]]) -> torch.Tensor:
    """Pad each current frame to (max_boxes, 7)."""
    max_boxes = max((s["boxes"].shape[0] for s in samples if s), default=0)
    B = len(samples)
    padded = torch.zeros((B, max_boxes, 7), dtype=torch.float32)

    for i, sample in enumerate(samples):
        if sample is None:
            continue
        arr = torch.from_numpy(sample["boxes"])
        num = arr.shape[0]
        padded[i, :num, :] = arr

    return padded

def _pad_histories(
    histories: List[List[Optional[Dict]]]
) -> torch.Tensor:
    """
    Pad each sample’s history:
      outputs (B, history_window, max_boxes, 7)
    """
    B = len(histories)
    H = max(len(h) for h in histories)
    max_boxes = max(
        (s["boxes"].shape[0] for h in histories for s in h if s), default=0
    )
    out = torch.zeros((B, H, max_boxes, 7), dtype=torch.float32)

    for i, hist in enumerate(histories):
        for t, sample in enumerate(hist):
            if sample is None:
                continue
            arr = torch.from_numpy(sample["boxes"])
            num = arr.shape[0]
            out[i, t, :num, :] = arr

    return out