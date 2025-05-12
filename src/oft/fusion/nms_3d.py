# src/oft/fusion/nms_3d.py
import yaml
from shapely.geometry import Polygon

# Konfiguration laden
config = yaml.safe_load(open('pipeline.yaml'))
iou_thresh = config['fusion']['iou_threshold']

def compute_bev_iou(box1, box2):
    """Berechnet IoU zweier 3D-Boxen in BEV."""
    poly1 = Polygon(box1.bev_corners())  # 4-Eck auf Bodenproj.
    poly2 = Polygon(box2.bev_corners())
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter/union if union>0 else 0

def non_max_suppression_3d(boxes):
    """Führt NMS auf einer Liste von 3D-Detektionen durch."""
    boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
    keep = []
    for box in boxes:
        # Box behalten, falls IoU mit allen behaltenen < Threshold
        if all(compute_bev_iou(box, kept) < iou_thresh for kept in keep):
            keep.append(box)
    return keep