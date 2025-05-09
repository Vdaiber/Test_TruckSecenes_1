#!/usr/bin/env python3
"""
3D-Box-Visualisierung: Corners, World→Camera, Projektion & Wireframe-Rendering.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from truckscenes import TruckScenes
from truckscenes.utils.geometry_utils import view_points, transform_matrix
from oft.utils.geometry import compute_box_corners



def world_to_camera(corners_w: np.ndarray,
                    ego_pose: dict,
                    calib: dict) -> np.ndarray:
    # unverändert
    pts = np.asarray(corners_w, dtype=np.float32)
    if pts.ndim == 2 and pts.shape == (8, 3):
        pts_w = pts.T
    elif pts.ndim == 2 and pts.shape[0] == 3:
        pts_w = pts
    else:
        raise ValueError(f"world_to_camera: unerwartete corners_w-Form {pts.shape}")
    ego_q = Quaternion(ego_pose["rotation"])
    ego_T = transform_matrix(ego_pose["translation"], ego_q, inverse=True)
    sens_q = Quaternion(calib["rotation"])
    sens_T = transform_matrix(calib["translation"], sens_q, inverse=True)
    H = sens_T @ ego_T
    homo = np.vstack((pts_w, np.ones((1, pts_w.shape[1]), dtype=np.float32)))
    return (H @ homo)[:3, :]

def render_boxes(ts: TruckScenes,
                 sample_token: str,
                 boxes: list,
                 img_in: np.ndarray = None,
                 sensor_channel: str = None,
                 color: tuple = (0, 255, 0),
                 thickness: int = 2) -> np.ndarray:
    """
    Zeichnet Wireframe-Boxen auf das Kamerabild des angegebenen Kanals.
    Lädt das Bild nun mit matplotlib.pyplot.imread() statt cv2.imread().
    """
    sample = ts.get("sample", sample_token)

    # --- 1) Basisbild vom Kamera-Kanal holen ---
    if img_in is None:
        if sensor_channel is None:
            raise ValueError("Kein camera_channel angegeben.")
        if sensor_channel not in sample["data"]:
            raise ValueError(
                f"Kamera-Kanal „{sensor_channel}“ nicht in sample.data: "
                f"{list(sample['data'].keys())}"
            )
        sd_tk = sample["data"][sensor_channel]
        sd    = ts.get("sample_data", sd_tk)
        fn    = sd["filename"]
        if not os.path.isabs(fn):
            fn = os.path.join(ts.dataroot, fn)

        # Matplotlib-Lader
        img = plt.imread(fn)
        # Falls float [0..1], in uint8 umwandeln
        if img.dtype in (np.float32, np.float64):
            img = (img * 255).astype(np.uint8)
        # matplot legt RGB an, OpenCV erwartet BGR für Zeichnungen
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    else:
        img = img_in.copy()

    # --- 2) Extrinsics laden ---
    sd    = ts.get("sample_data", sd_tk)
    calib = ts.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego   = ts.get("ego_pose",         sd["ego_pose_token"])
    K     = np.array(calib.get("camera_intrinsic", np.eye(3)),
                     dtype=np.float32).reshape(3, 3)

    # --- 3) Jede Box projizieren und zeichnen ---
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]

    for box in boxes:
        # Ecken in Weltkoordinaten
        if hasattr(box, "corners"):
            cw = box.corners()
        else:
            cw = compute_box_corners(box)

        # in Kamera-Koordinaten
        cc = world_to_camera(cw, ego, calib)

        # Projektion & Drahtgitter
        pts3 = view_points(cc, K, normalize=True)
        pts2 = pts3[:2, :].T.astype(np.int32)
        for i, j in edges:
            cv2.line(img, tuple(pts2[i]), tuple(pts2[j]), color, thickness)

    return img