#!/usr/bin/env python3
"""
Debug-Rendering einer einzelnen Ground-Truth-Box:
  1) lädt eine Szene
  2) nimmt die erste Annotation
  3) holt die 8 Box-Ecken direkt aus box_obj.corners()
  4) transformiert sie per Kamera-Extrinsic und Intrinsic ins Bild
  5) zeichnet rote Punkte für jede Ecke
  6) speichert unter /output/debug_<token>.jpg
"""
#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np

# Damit wir src/ und examples/ importieren können
BASE = os.getcwd()
sys.path.insert(0, os.path.join(BASE, "src"))

from truckscenes import TruckScenes
from oft.utils.visualization import (
    compute_box_corners,
    world_to_camera,
    render_boxes
)

def main(dataroot, version, output_dir, ann_token=None):
    ts = TruckScenes(version=version, dataroot=dataroot)
    os.makedirs(output_dir, exist_ok=True)

    # harte Kodierung: die Annotation, die wir debuggen wollen
    if ann_token is None:
        ann_token = "50873dd5195b47bc9544ac2821e054e0"

    # 1) Hole Annotation und Box
    ann = ts.get("sample_annotation", ann_token)
    box = ts.get_box(ann_token)

    # 2) Guard-Check
    if len(box) != 7:
        raise ValueError(f"Sample-Box erwartet 7 Werte, got {len(box)}")

    # 3) Compute Corners
    corners_w = compute_box_corners(box)  # (3×8)

    # 4) Kamera-Infos
    sample      = ts.get("sample", ann["sample_token"])
    first_data  = ts.get("sample_data", sample["data"]["CAMERA_LEFT_FRONT"])
    K           = np.array(ts.get("calibrated_sensor", first_data["calibrated_sensor_token"])["camera_intrinsic"])
    sensor_cal  = ts.get("calibrated_sensor", first_data["calibrated_sensor_token"])
    ego_pose    = ts.get("ego_pose", first_data["ego_pose_token"])

    # 5) Projektions-Test
    corners_c = world_to_camera(corners_w, ego_pose, sensor_cal)
    print("Projizierte 2D-Ecken:\n", corners_c)

    # 6) Optional: mit render_boxes zeichnen
    img = render_boxes(ts, ann["sample_token"], [box], img_in=None)
    if img is None or not hasattr(img, "shape") or img.size == 0:
        img = np.zeros((720, 1280, 3), dtype=np.uint8)

    out_path = os.path.join(output_dir, f"debug_{ann_token}.jpg")
    cv2.imwrite(out_path, img)
    print("Debug-Bild gespeichert unter", out_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataroot")
    parser.add_argument("version")
    parser.add_argument("output_dir")
    parser.add_argument("--ann_token", type=str, default=None)
    args = parser.parse_args()
    main(args.dataroot, args.version, args.output_dir, args.ann_token)