#!/usr/bin/env python3
"""
3D-Box-Visualisierung für TruckScenes (Tutorial-konform).

Draws wireframe on camera image using DevKit-Box-Objekte,
die über ts.get_sample_data() schon in Kamerakoordinaten kommen.
"""

import os
import cv2
import numpy as np
from truckscenes.utils.colormap       import get_colormap
from truckscenes.utils.geometry_utils import view_points, BoxVisibility

# DevKit-Colormap: category_name → (R,G,B)
CLASS_COLORS = get_colormap()

# die 12 Wireframe-Kanten
_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

def draw_boxes_on_image(
    image: np.ndarray,
    boxes: list,
    K: np.ndarray,
    z_threshold: float = 0.0,
    thickness: int = 2
) -> np.ndarray:
    """
    Zeichnet Wireframe-Boxen, Box-Ecken sind bereits in Kamera-Koords:
      – image: BGR OpenCV Image
      – boxes: Liste von truckscenes.utils.data_classes.Box
      – K:      3×3 Kamera-Intrinsisch
      – z_threshold: minimale Kamera-Tiefe (m) eines Eckpunkts
      – thickness:   Linienbreite (px)
    """
    img = image.copy()
    for box in boxes:
        corners_cam = box.corners()  # (3,8) in Kamerakoordinate
        pts = view_points(corners_cam, K, normalize=True)  # (3,8)
        xs, ys, zs = pts

        # wenn alle Eckpunkte hinter dem Threshold liegen → skip
        if np.max(zs) <= z_threshold:
            continue

        # Farbe aus DevKit-Colormap (Kategorie → RGB)
        rgb   = CLASS_COLORS.get(box.name, (0,255,0))
        color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))  # BGR für OpenCV

        # Drahtgitter zeichnen
        for i,j in _EDGES:
            p1 = (int(xs[i]), int(ys[i]))
            p2 = (int(xs[j]), int(ys[j]))
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

    return img


def render_sample_boxes(
    ts,
    sample_token: str,
    sensor_channel: str,
    visibility: BoxVisibility = BoxVisibility.ANY,
    z_threshold: float     = 0.0,
    thickness: int         = 2,
    out_path: str          = None
) -> np.ndarray:
    """
    – Lädt das Kamerabild für sample_token & sensor_channel.
    – Holt GT-Boxen per DevKit get_sample_data(..., box_vis_level=visibility).
    – Zeichnet Wireframe via draw_boxes_on_image().
    """

    # --- Bild laden ---
    sample = ts.get("sample", sample_token)
    if sensor_channel not in sample["data"]:
        raise ValueError(f"Sensor-Channel '{sensor_channel}' nicht in sample.data")
    sd_tk = sample["data"][sensor_channel]
    sd    = ts.get("sample_data", sd_tk)
    fn    = sd["filename"]
    if not os.path.isabs(fn):
        fn = os.path.join(ts.dataroot, fn)
    img = cv2.imread(fn, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {fn}")

    # --- Intrinsic holen ---
    calib = ts.get("calibrated_sensor", sd["calibrated_sensor_token"])
    K     = np.array(calib["camera_intrinsic"],
                     dtype=np.float32).reshape(3,3)

    # --- GT-Boxen holen & filtern ---
    _, boxes, _ = ts.get_sample_data(
        sd_tk,
        box_vis_level=visibility
    )

    # --- Drahtgitter zeichnen ---
    img_out = draw_boxes_on_image(
        img, boxes, K,
        z_threshold=z_threshold,
        thickness=thickness
    )

    # --- speichern / zurückgeben ---
    if out_path:
        cv2.imwrite(out_path, img_out)
    return img_out