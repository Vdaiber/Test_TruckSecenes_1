#!/usr/bin/env python3
import os
import sys
import yaml
import json
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# add project src/ in PYTHONPATH
HERE     = os.path.abspath(__file__)
SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from oft.utils.sensor_utils import (
    get_camera_intrinsic,
    get_sensor_extrinsic,
    draw_boxes_on_image,
    CLASS_COLORS,
)
from truckscenes.utils.geometry_utils import view_points
from oft.data.dataset import TruckScenesDataset

def load_config(path="config/pipeline.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_fusion_detections(path, sample_token, cam_ch):
    """
    Erwartet eine JSON-Liste von Einträgen
      {'sample_token': str,
       'dets': {
           'CAMERA_LEFT_FRONT': [
               {'translation':[x,y,z],'wlh':[w,l,h],'yaw':θ}, ...
           ], ...
       }}
    """
    with open(path, "r") as f:
        all_d = json.load(f)
    for entry in all_d:
        if entry["sample_token"] == sample_token:
            return entry["dets"].get(cam_ch, [])
    return []

def project_and_filter(boxes, K, H, img_shape):
    """
    Zeichnet nur die Boxen, deren Eckpunkte sichtbar sind,
    und liefert ihre 2D-Projektionen plus 3D-Mittelpunkt.
    """
    H_img, W_img = img_shape[:2]
    visible = []
    for box in boxes:
        # 3D-Corners und Homo
        corners = box.corners()  # (3,8)
        homo    = np.vstack((corners, np.ones((1,8),dtype=np.float32)))  # (4,8)
        cam_pts = H @ homo                                                   # (4,8)
        zs      = cam_pts[2]
        if not np.all(zs>0):  # weg damit, wenn eine Ecke hinter der Kamera
            continue
        pts2d = view_points(cam_pts[:3], K, normalize=True)  # (3,8)
        xs, ys = pts2d[0], pts2d[1]
        if np.any((xs>=0)&(xs<W_img)&(ys>=0)&(ys<H_img)):
            # speichere primär den Mittelpunkt
            center_cam = (float(cam_pts[0].mean()), float(cam_pts[1].mean()), float(cam_pts[2].mean()))
            visible.append((box, pts2d, center_cam))
    return visible

def main():
    cfg = load_config()
    dcfg = cfg["dataset"]
    vcfg = cfg["visualization"]
    ocfg = cfg["output"]
    rcfg = cfg["render"]

    # sample index aus config
    sample_i  = vcfg.get("sample_idx", 0)
    cam_ch    = vcfg["camera_channel"]
    thickness = rcfg["line_thickness"]
    fusion_fn = ocfg["dets_json"]
    out_dir   = ocfg["noise_comparison_dir"].replace("noise_comparison", "fusion_comparison")
    os.makedirs(out_dir, exist_ok=True)

    # Dataset-Instanzen (nur GT, kein Noise hier)
    common = dict(
        dataroot=dcfg["dataroot"],
        version=dcfg["version"].strip(),
        history_window=0,
        max_boxes=dcfg.get("gt_max_boxes") or 0,
        augment_noise_std=0.0
    )
    ds = TruckScenesDataset(**common)

    if not (0<=sample_i<len(ds)):
        raise IndexError(f"sample_idx {sample_i} out of range")

    # Sample‐Token und Bild laden
    sample_token = ds.samples[sample_i]
    samp         = ds.ts.get("sample", sample_token)
    sd_tok       = samp["data"][cam_ch]
    sd           = ds.ts.get("sample_data", sd_tok)
    img_fn       = sd["filename"]
    if not os.path.isabs(img_fn):
        img_fn = os.path.join(ds.ts.dataroot, img_fn)
    img = cv2.imread(img_fn)
    if img is None:
        raise FileNotFoundError(f"could not read {img_fn}")

    # Intrinsic + Extrinsic
    calib = ds.ts.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego   = ds.ts.get("ego_pose",       sd["ego_pose_token"])
    K     = get_camera_intrinsic(calib)
    H     = get_sensor_extrinsic(ego, calib)

    # 1) Ground‐Truth‐Boxen
    gt_boxes = [ds.ts.get_box(a) for a in samp["anns"]]
    vis_gt   = project_and_filter(gt_boxes, K, H, img.shape)

    # 2) Gefusete Boxen aus JSON
    dets = load_fusion_detections(fusion_fn, sample_token, cam_ch)
    # wandle um in Box‐Objekte
    from truckscenes.utils.data_classes import Box
    fusion_boxes = []
    for d in dets:
        b = Box(
            name="fused",
            translation=d["translation"],
            wlh=d["wlh"],
            rotation=[0,0,0,1],  # yaw-only in wlh/yaw
        )
        # override yaw manually:
        b.orientation = float(d["yaw"])
        fusion_boxes.append(b)
    vis_fus = project_and_filter(fusion_boxes, K, H, img.shape)

    # 3) Hungarian Matching auf Mittelpunkten
    if vis_gt and vis_fus:
        gt_centers  = np.array([c for (_,_,c) in vis_gt])
        fu_centers  = np.array([c for (_,_,c) in vis_fus])
        costmat     = np.linalg.norm(
            gt_centers[:,None,:] - fu_centers[None,:,:],
            axis=-1
        )  # (Ng, Nf)
        row, col   = linear_sum_assignment(costmat)
        matches    = [(r,c) for r,c in zip(row,col)]
    else:
        matches = []

    # 4) Draw
    orig = CLASS_COLORS.copy()

    # GT grün
    for k in CLASS_COLORS: CLASS_COLORS[k] = (0,255,0)
    out = img.copy()
    for box, pts2d, _ in vis_gt:
        out = draw_boxes_on_image(out, [box], K, H, thickness)

    # Fusion rot
    CLASS_COLORS.clear(); CLASS_COLORS.update(orig)
    for k in CLASS_COLORS: CLASS_COLORS[k] = (0,0,255)
    for box, pts2d, _ in vis_fus:
        out = draw_boxes_on_image(out, [box], K, H, thickness)

    # Linien zwischen Matches
    for i_f, j_f in matches:
        _, _, c1 = vis_gt[i_f]
        _, _, c2 = vis_fus[j_f]
        # projiziere Mittelpunkte nochmal
        p1 = view_points(np.array(c1).reshape(3,1), K, normalize=True)[:2,0]
        p2 = view_points(np.array(c2).reshape(3,1), K, normalize=True)[:2,0]
        cv2.line(out,
                 (int(p1[0]),int(p1[1])),
                 (int(p2[0]),int(p2[1])),
                 (255,255,0),
                 thickness, cv2.LINE_AA)

    # 5) Speichern
    fn = os.path.join(out_dir, f"sample_{sample_i:03d}_fusion_vs_gt.jpg")
    cv2.imwrite(fn, out)
    print("wrote", fn)


if __name__=="__main__":
    main()