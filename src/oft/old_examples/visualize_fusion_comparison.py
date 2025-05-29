#!/usr/bin/env python3
"""
example/visualize_fusion_comparison.py

Vergleicht GT-Wireframes mit Fusion-Detections auf dem Bild.
"""
import argparse
import os
import sys
import yaml
import json
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from truckscenes.utils.data_classes import Box
from truckscenes.utils.geometry_utils import view_points

from oft.utils.config import load_config
from oft.data.old_dataset import TruckScenesDataset
from oft.utils.sensor_utils import (
    get_camera_intrinsic,
    get_sensor_extrinsic,
    draw_boxes_on_image,
    CLASS_COLORS
)

def load_fusion_detections(path, sample_token, cam_ch):
    all_data = json.load(open(path, "r"))
    if isinstance(all_data, dict):
        all_data = [
            {"sample_token": tok, "dets": {cam_ch: all_data[tok]}}
            for tok in all_data
        ]
    for entry in all_data:
        if entry["sample_token"] == sample_token:
            return entry.get("dets", {}).get(cam_ch, [])
    return []

def project_and_filter(boxes, K, H, img_shape):
    H_img, W_img = img_shape[:2]
    vis = []
    for box in boxes:
        corners = box.corners()                         # (3,8)
        homo    = np.vstack((corners, np.ones((1,8))))
        cam_pts = H @ homo                              # (4,8)
        if not np.all(cam_pts[2] > 0):
            continue
        pts2d = view_points(cam_pts[:3], K, normalize=True)
        xs, ys = pts2d[0], pts2d[1]
        if np.any((xs>=0)&(xs<W_img)&(ys>=0)&(ys<H_img)):
            center = (
                float(cam_pts[0].mean()),
                float(cam_pts[1].mean()),
                float(cam_pts[2].mean())
            )
            vis.append((box, pts2d, center))
    return vis

def main():
    # ---- Argumente & Config ----
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", default="config/pipeline.yaml",
                   help="Pfad zu config/pipeline.yaml")
    args = p.parse_args()

    cfg  = load_config(args.config)
    dcfg = cfg["dataset"]
    vcfg = cfg["visualization"]
    ocfg = cfg["output"]
    rcfg = cfg["render"]

    sample_i = vcfg.get("sample_idx", 0)
    cam_ch   = vcfg["camera_channel"]
    out_dir  = ocfg["fusion_comparison_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # ---- Dataset für GT ----
    ds = TruckScenesDataset(
        dataroot        = dcfg["dataroot"],
        version         = dcfg["version"].strip(),
        history_window  = 0,
        max_boxes       = dcfg.get("gt_max_boxes") or 0,
        augment_noise_std = 0.0
    )
    ts = ds.ts

    if not (0 <= sample_i < len(ds)):
        raise IndexError(f"sample_idx {sample_i} out of range")

    sample_token = ds.samples[sample_i]
    samp = ts.get("sample", sample_token)
    sd   = ts.get("sample_data", samp["data"][cam_ch])

    img_fn = sd["filename"]
    if not os.path.isabs(img_fn):
        img_fn = os.path.join(dcfg["dataroot"], img_fn)
    img = cv2.imread(img_fn)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_fn}")

    # Kamera-Matrizen
    calib = ts.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego   = ts.get("ego_pose",       sd["ego_pose_token"])
    K     = get_camera_intrinsic(calib)
    H     = get_sensor_extrinsic(ego, calib)

    # 1) GT-Boxen mit DevKit holen
    _, gt_boxes, _ = ts.get_sample_data(
        samp["data"][cam_ch],
        box_vis_level=rcfg["box_visibility"]
    )

    # 2) Fusion-Boxen aus JSON laden und als Box-Objekte anlegen
    dets = load_fusion_detections(ocfg["dets_json"], sample_token, cam_ch)
    fusion_boxes = []
    for d in dets:
        tx, ty, tz = d["translation"]
        w, l, h     = d["wlh"]
        yaw         = d["yaw"]
        # Quaternion aus yaw
        qx, qy, qz, qw = 0.0, 0.0, np.sin(yaw/2), np.cos(yaw/2)
        fusion_boxes.append(
            Box(
                center      = [tx, ty, tz],
                size        = [w, l, h],
                orientation = (qx, qy, qz, qw)
            )
        )

    # 3) Project & filter visibility
    vis_gt  = project_and_filter(gt_boxes,   K, H, img.shape)
    vis_fus = project_and_filter(fusion_boxes, K, H, img.shape)
    print(f"[DEBUG] Visible GT: {len(vis_gt)}, Fusion: {len(vis_fus)}")

    # 4) Zeichnen
    out = img.copy()
    orig_colors = CLASS_COLORS.copy()

    # Grün für GT
    for k in CLASS_COLORS: CLASS_COLORS[k] = (0,255,0)
    out = draw_boxes_on_image(
        out, [b for b,_,_ in vis_gt], K, H, rcfg["line_thickness"]
    )

    # Rot für Fusion
    CLASS_COLORS.clear(); CLASS_COLORS.update(orig_colors)
    for k in CLASS_COLORS: CLASS_COLORS[k] = (0,0,255)
    out = draw_boxes_on_image(
        out, [b for b,_,_ in vis_fus], K, H, rcfg["line_thickness"]
    )

    # 5) Matching-Linien (Hungarian)
    if vis_gt and vis_fus:
        gt_centers = np.array([c for _,_,c in vis_gt])
        fu_centers = np.array([c for _,_,c in vis_fus])
        cost = np.linalg.norm(gt_centers[:,None,:] - fu_centers[None,:,:], axis=-1)
        row, col = linear_sum_assignment(cost)
        for i,j in zip(row,col):
            p1 = view_points(np.array(vis_gt[i][2]).reshape(3,1), K, normalize=True)[:2,0]
            p2 = view_points(np.array(vis_fus[j][2]).reshape(3,1), K, normalize=True)[:2,0]
            cv2.line(
                out,
                (int(p1[0]),int(p1[1])),
                (int(p2[0]),int(p2[1])),
                (255,255,0),
                rcfg["line_thickness"],
                cv2.LINE_AA
            )

    # 6) Abspeichern
    fn = os.path.join(out_dir, f"{sample_i:03d}_{sample_token}_fusion_vs_gt.jpg")
    cv2.imwrite(fn, out)
    print(f"✓ Fusion-Comparison saved to {fn}")

if __name__=="__main__":
    main()