#!/usr/bin/env python3
import os
import cv2
import numpy as np
from pyquaternion import Quaternion
from truckscenes.utils.colormap import get_colormap
from truckscenes.utils.geometry_utils import view_points, transform_matrix
from truckscenes.utils.data_classes import Box

CLASS_COLORS = get_colormap()

def get_camera_intrinsic(calib: dict) -> np.ndarray:
    K = np.array(calib["camera_intrinsic"], dtype=np.float32).reshape(3, 3)
    return K

def get_sensor_extrinsic(ego_pose: dict, calib: dict) -> np.ndarray:
    """
    Berechnet 4×4 Welt→Sensor-Extrinsik:
    invertiertes Ego↔World und invertiertes Ego↔Sensor.
    """
    q_e = Quaternion(ego_pose["rotation"])
    T_e = transform_matrix(ego_pose["translation"], q_e, inverse=True)
    q_s = Quaternion(calib["rotation"])
    T_s = transform_matrix(calib["translation"], q_s, inverse=True)
    return T_s @ T_e

def draw_boxes_on_image(
    image: np.ndarray,
    boxes: list,
    sensor_intrinsic: np.ndarray,
    sensor_extrinsic: np.ndarray,
    thickness: int = 2
) -> np.ndarray:
    img = image.copy()
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    for box in boxes:
        corners_w = box.corners()                          # (3,8)
        homo = np.vstack((corners_w, np.ones((1,8),dtype=np.float32)))
        pts_cam = sensor_extrinsic @ homo                  # (4,8)
        pts_cam = pts_cam[:3,:]
        pts_img = view_points(pts_cam, sensor_intrinsic, normalize=True)
        xs, ys, zs = pts_img[0], pts_img[1], pts_img[2]
        if not np.all(zs > 0):
            continue
        rgb   = CLASS_COLORS.get(box.name, (0,255,0))
        color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        for i,j in edges:
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
    sample = ts.get("sample", sample_token)
    if sensor_channel not in sample["data"]:
        raise ValueError(f"Kanal '{sensor_channel}' nicht gefunden")
    sd = ts.get("sample_data", sample["data"][sensor_channel])
    fn = sd["filename"]
    if not os.path.isabs(fn):
        fn = os.path.join(ts.dataroot, fn)
    img = cv2.imread(fn, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {fn}")

    calib = ts.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego   = ts.get("ego_pose",        sd["ego_pose_token"])
    K     = get_camera_intrinsic(calib)
    H     = get_sensor_extrinsic(ego, calib)

    ann_tokens = sample["anns"]
    boxes = [ts.get_box(a) for a in ann_tokens]

    img_out = draw_boxes_on_image(img, boxes, K, H, thickness)
    if out_path:
        cv2.imwrite(out_path, img_out)
    return img_out