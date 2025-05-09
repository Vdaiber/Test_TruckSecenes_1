"""
3D Non-Maximum Suppression (BEV IoU) for TruckScenes detections.
"""

# src/oft/fusion/nms_3d.py
import torch
from ..utils.geometry import box3d_iou_torch

def nms_3d(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.3) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    sorted_scores, idxs = torch.sort(scores, descending=True)
    sorted_boxes = boxes[idxs]
    keep = []
    while sorted_boxes.size(0):
        i = idxs[0].item()
        keep.append(i)
        if sorted_boxes.size(0) == 1:
            break
        ious = box3d_iou_torch(sorted_boxes[0:1], sorted_boxes[1:])
        mask = ious <= iou_threshold
        idxs = idxs[1:][mask]
        sorted_boxes = sorted_boxes[1:][mask]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)