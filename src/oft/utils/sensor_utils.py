#!/usr/bin/env python3
"""
3D-Box Visualisierung für TruckScenes

Enthält:
- get_camera_intrinsic(calib)
- get_sensor_extrinsic(ego_pose, calib)
- draw_boxes_on_image(image, boxes, K, H, thickness)
- render_sample_boxes(ts, sample_token, sensor_channel, thickness, out_path)
"""

import os
import cv2
import numpy as np
from pyquaternion import Quaternion

from truckscenes.utils.colormap import get_colormap
from truckscenes.utils.geometry_utils import view_points, transform_matrix
from truckscenes.utils.data_classes import Box  # nur für Typen und .corners()/.name

# Default-Colormap: category_name → (R,G,B)
CLASS_COLORS = get_colormap()


def get_camera_intrinsic(calib: dict) -> np.ndarray:
    """
    Extrahiert die 3×3 Kameraintrinsische Matrix aus dem 'calibrated_sensor'-Record.
    """
    K = np.array(calib["camera_intrinsic"], dtype=np.float32).reshape(3, 3)
    return K


def get_sensor_extrinsic(ego_pose: dict, calib: dict) -> np.ndarray:
    """
    Berechnet die 4×4-Transformationsmatrix vom Welt- in das Kamerakoordinatensystem.
    world→ego (invertiert) gefolgt von ego→sensor (invertiert).
    """
    # 1) world→ego
    q_e = Quaternion(ego_pose["rotation"])
    T_e = transform_matrix(ego_pose["translation"], q_e, inverse=True)
    # 2) ego→sensor
    q_s = Quaternion(calib["rotation"])
    T_s = transform_matrix(calib["translation"], q_s, inverse=True)
    # 3) zusammengesetzt:
    return T_s @ T_e


def draw_boxes_on_image(
    image: np.ndarray,
    boxes: list,
    sensor_intrinsic: np.ndarray,
    sensor_extrinsic: np.ndarray,
    thickness: int = 2
) -> np.ndarray:
    """
    Zeichnet 3D-Drahtgitter-Boxen auf das Bild.

    Args:
      - image: H×W×3 BGR-Array (OpenCV)
      - boxes: Liste von truckscenes.utils.data_classes.Box
      - sensor_intrinsic: 3×3 Matrix
      - sensor_extrinsic: 4×4 Welt→Kamera Matrix
      - thickness: Linienbreite
    Returns:
      - Annotiertes Bild (NumPy-Array)
    """
    img = image.copy()
    # Drahtgitter-Kanten
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    for box in boxes:
        # 1) Ecken im Welt-Koordinatensystem: (3,8)
        corners_w = box.corners()
        # 2) Homogene Koordinaten world→cam: (4,8)
        homo = np.vstack((corners_w, np.ones((1, corners_w.shape[1]), dtype=np.float32)))
        pts_cam = sensor_extrinsic @ homo      # (4,8) → (3,8)
        pts_cam = pts_cam[:3, :]
        # 3) Projektion ins Bild
        pts_img = view_points(pts_cam, sensor_intrinsic, normalize=True)  # (3,8)
        xs, ys, zs = pts_img[0, :], pts_img[1, :], pts_img[2, :]
        # nur Boxen, die komplett in Front sind
        if not np.all(zs > 0):
            continue
        # Farbe per Kategorie (RGB→BGR)
        rgb   = CLASS_COLORS.get(box.name, (0, 255, 0))
        color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        # 4) Linien zeichnen
        for i, j in edges:
            p1 = (int(xs[i]), int(ys[i]))
            p2 = (int(xs[j]), int(ys[j]))
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    return img


def render_sample_boxes(
    ts,
    sample_token: str,
    sensor_channel: str,
    thickness: int = 2,
    out_path: str = None
) -> np.ndarray:
    """
    Convenience: Lädt Bild + GT-Boxen aus TruckScenes, ruft draw_boxes_on_image und speichert.

    Args:
      - ts: TruckScenes-Instanz
      - sample_token: Sample-Token
      - sensor_channel: z.B. 'CAMERA_LEFT_FRONT'
      - thickness: Linienbreite (aus pipeline.yaml)
      - out_path: optionaler Pfad zum Speichern (z.B. 'out.jpg')
    Returns:
      - Annotiertes Bild (NumPy-Array)
    """
    sample = ts.get("sample", sample_token)

    # --- Bild laden ---
    if sensor_channel not in sample["data"]:
        raise ValueError(f"Kanal '{sensor_channel}' nicht in sample.data")
    sd_tk = sample["data"][sensor_channel]
    sd    = ts.get("sample_data", sd_tk)
    fn    = sd["filename"]
    if not os.path.isabs(fn):
        fn = os.path.join(ts.dataroot, fn)
    img = cv2.imread(fn, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {fn}")

    # --- Intrinsic & Extrinsic berechnen ---
    calib = ts.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego   = ts.get("ego_pose", sd["ego_pose_token"])
    K     = get_camera_intrinsic(calib)
    H     = get_sensor_extrinsic(ego, calib)

    # --- GT-Boxen holen ---
    ann_tokens = sample["anns"]
    boxes      = [ts.get_box(a) for a in ann_tokens]

    # --- Draufmalen & Speichern ---
    img_out = draw_boxes_on_image(img, boxes, K, H, thickness)
    if out_path:
        cv2.imwrite(out_path, img_out)
    return img_out