#!/usr/bin/env python3
import numpy as np
from oft.fusion.nms_3d import nms_bev_3d

def make_box(x):
    """
    Hilfsfunktion: baut eine 3D-Box auf
    Format: [x, y, z, length, width, height, yaw]
    """
    return np.array([x, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0], dtype=np.float32)

def main():
    # drei Test-Boxen: zwei nahe beieinander, eine weit entfernt
    boxes = np.vstack([
        make_box(0.0),
        make_box(0.5),   # stark überlappend mit Box 0.0
        make_box(10.0),  # weit weg
    ])
    scores = np.array([0.9, 0.8, 0.5], dtype=np.float32)

    keep = nms_bev_3d(boxes, scores, iou_threshold=0.3)
    print("Beibehaltene Indizes nach NMS:", keep)
    # Erwartung: [0, 2] — die Box bei x=0.5 wird wegen IoU>0.3 zu Box 0 unterdrückt

if __name__ == "__main__":
    main()