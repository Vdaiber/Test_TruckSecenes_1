# Datei: nms_3d.py

import numpy as np
from shapely.geometry import Polygon

def box_to_polygon(box):
    """
    Konvertiert eine 3D-Box (Center x,y,z, Länge, Breite, Höhe, Yaw) in ein 2D-Polygon (BEV).
    Erwartetes Format von box: [x, y, z, length, width, height, yaw]
    """
    x, y, _, length, width, _, yaw = box
    # Eckpunkte der Box vor Rotation (z-Achse)
    dx = length / 2
    dy = width  / 2
    corners = np.array([
        [ dx,  dy],
        [ dx, -dy],
        [-dx, -dy],
        [-dx,  dy]
    ])
    # Rotation um die z-Achse (Yaw)
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    rot_mat = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    corners_rot = corners.dot(rot_mat.T)
    # Translation zum Boxzentrum
    corners_rot[:, 0] += x
    corners_rot[:, 1] += y
    return Polygon(corners_rot)

def bev_iou(box1, box2):
    """
    Berechnet die Intersection-over-Union (IoU) zweier 3D-Boxen in der Bodenebene (BEV).
    Die Boxen werden dabei als 2D-Polygone interpretiert. IoU = Überlappungsfläche / Vereinigungsfläche.
    """
    poly1 = box_to_polygon(box1)
    poly2 = box_to_polygon(box2)
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter_area = poly1.intersection(poly2).area
    union_area = poly1.area + poly2.area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def nms_bev_3d(boxes, scores, iou_threshold=0.5):
    """
    Führt eine BEV-basierte NMS auf 3D-Boxen durch.
    
    Argumente:
        boxes (np.ndarray): Array der Form (N,7) mit [x,y,z,length,width,height,yaw].
        scores (np.ndarray): Konfidenzwerte der Boxen (N,). Sortierung erfolgt absteigend.
        iou_threshold (float): IoU-Schwelle für das Entfernen überlappender Boxen.
    
    Rückgabe:
        keep_indices (List[int]): Indizes der Boxen, die nach NMS beibehalten werden.
    """
    if len(boxes) == 0:
        return []
    # Sortiere Boxen nach Score (höchster zuerst)
    order = np.argsort(scores)[::-1]
    keep = []
    for idx in order:
        keep_flag = True
        for kept_idx in keep:
            iou = bev_iou(boxes[idx], boxes[kept_idx])
            if iou > iou_threshold:
                keep_flag = False
                break
        if keep_flag:
            keep.append(idx)
    return keep

if __name__ == "__main__":
    # Beispielnutzung / Test
    boxes = np.array([
        [0, 0, 0, 4, 2, 1.5, 0.0],
        [1, 0, 0, 4, 2, 1.5, 0.0],
        [10, 0, 0, 3, 1.5, 1.5, 0.0]
    ])
    scores = np.array([0.9, 0.8, 0.5])
    keep = nms_bev_3d(boxes, scores, iou_threshold=0.3)
    print("Beibehaltene Indizes nach NMS:", keep)