import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ..utils.geometry import box3d_iou_torch

class HungarianTracker:
    """IoU-basierter Multi-Object-Tracker mit Birth/Death-Handling."""

    def __init__(self, max_age: int = 3, iou_threshold: float = 0.25):
        self.tracks      = []       # Liste der aktuellen Tracks
        self.next_id     = 0
        self.max_age     = max_age
        self.iou_thr     = iou_threshold
        self.cost_thr    = 1.0 - iou_threshold

    def update(self, detections: np.ndarray) -> list:
        M = len(self.tracks)
        N = len(detections)

        # 1) Keine Tracks → alle Detections neu anlegen
        if M == 0:
            for det in detections:
                self.tracks.append({'id':self.next_id, 'box':det, 'age':0})
                self.next_id += 1
            return list(self.tracks)

        # 2) Kostenmatrix = 1 − IoU
        cost = np.zeros((M, N), dtype=np.float32)
        for i, tr in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou = box3d_iou_torch(
                    torch.from_numpy(tr['box']).float().unsqueeze(0),
                    torch.from_numpy(det).float().unsqueeze(0)
                ).item()
                cost[i, j] = 1.0 - iou

        # 3) Assignment
        row_idx, col_idx = linear_sum_assignment(cost)
        assigned = set()
        updated  = []

        # 4) Matched-Update (nur wenn cost < cost_thr)
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < self.cost_thr:
                tr = self.tracks[r]
                tr['box'] = detections[c]
                tr['age'] = 0
                updated.append(tr)
                assigned.add(c)

        # 5) Neue Tracks für unmatched detections
        for j, det in enumerate(detections):
            if j not in assigned:
                updated.append({'id':self.next_id, 'box':det, 'age':0})
                self.next_id += 1

        # 6) Altern für nicht upgedatete Tracks
        existing_boxes = [t['box'] for t in updated]
        for tr in self.tracks:
            if not any(np.array_equal(tr['box'], b) for b in existing_boxes):
                tr['age'] += 1
                if tr['age'] < self.max_age:
                    updated.append(tr)

        self.tracks = updated
        return list(self.tracks)