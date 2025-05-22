#!/usr/bin/env python3
import os, sys, math, json, cv2, numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image
from pyquaternion import Quaternion  # neu

# ensure repo/src on PYTHONPATH
HERE     = os.path.abspath(__file__)
SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from oft.utils.config import load_config
from oft.data.dataset import TruckScenesDataset
from truckscenes import TruckScenes
from truckscenes.utils.geometry_utils import view_points
from oft.utils.sensor_utils import get_camera_intrinsic, get_sensor_extrinsic, draw_boxes_on_image, CLASS_COLORS
from truckscenes.utils.data_classes import Box

def try_load_image(path):
    img = cv2.imread(path)
    if img is not None:
        return img
    pil = Image.open(path)
    arr = np.array(pil)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr

def main():
    cfg   = load_config("config/pipeline.yaml")
    dcfg  = cfg["dataset"]
    vcfg  = cfg["visualization"]
    rcfg  = cfg["render"]
    ocfg  = cfg["output"]

    sample_i = vcfg.get("sample_idx", 0)
    cam_ch   = vcfg["camera_channel"]

    # Dataset → nur zum GT abrufen
    ds = TruckScenesDataset(
        dataroot         = dcfg["dataroot"],
        version          = dcfg["version"].strip(),
        history_window   = 0,
        max_boxes        = dcfg.get("gt_max_boxes") or 0,
        augment_noise_std= 0.0
    )
    assert 0 <= sample_i < len(ds), "sample_idx out of range"

    token = ds.samples[sample_i]
    ts    = ds.ts

    # Bildpfad
    samp   = ts.get("sample", token)
    sd     = ts.get("sample_data", samp["data"][cam_ch])
    img_fn = sd["filename"]
    if not os.path.isabs(img_fn):
        img_fn = os.path.join(dcfg["dataroot"], img_fn)

    img = try_load_image(img_fn)
    K   = get_camera_intrinsic(ts.get("calibrated_sensor", sd["calibrated_sensor_token"]))
    H   = get_sensor_extrinsic(ts.get("ego_pose", sd["ego_pose_token"]),
                               ts.get("calibrated_sensor", sd["calibrated_sensor_token"]))

    # 1) Ground-Truth per Annotation
    gt_boxes = []
    for ann in samp["anns"]:
        gt_boxes.append(ts.get_box(ann))

    # 2) Fusion-Detections
    raw = json.load(open(ocfg["dets_json"], "r"))
    if isinstance(raw, dict):
        raw = [{"sample_token":k, "dets":{cam_ch:v}} for k,v in raw.items()]
    dets = next((e["dets"].get(cam_ch,[]) for e in raw if e["sample_token"]==token), [])

    fus_boxes = []
    for d in dets:
        t = d["translation"]
        s = d["wlh"]
        yaw = d["yaw"]
        # Quaternion erzeugen
        qx, qy = 0.0, 0.0
        qz = math.sin(yaw/2)
        qw = math.cos(yaw/2)
        q = Quaternion(qw, qx, qy, qz)
        fus_boxes.append(Box(
            center      = [float(x) for x in t],
            size        = [float(x) for x in s],
            orientation = q
        ))

    # 3) Projektion + Filterung
    def project_filter(box_list):
        vis = []
        H_img, W_img = img.shape[:2]
        for box in box_list:
            corners = box.corners()                   # (3,8)
            homo    = np.vstack((corners, np.ones((1,8),dtype=np.float32)))
            cam_pts = H @ homo                       # (4,8)
            if not np.all(cam_pts[2]>0):
                continue
            pts2d = view_points(cam_pts[:3], K, normalize=True)
            xs, ys = pts2d[0], pts2d[1]
            if np.any((xs>=0)&(xs<W_img)&(ys>=0)&(ys<H_img)):
                center = (cam_pts[0].mean(), cam_pts[1].mean(), cam_pts[2].mean())
                vis.append((box, pts2d, center))
        return vis

    gt_vis  = project_filter(gt_boxes)
    fus_vis = project_filter(fus_boxes)

    # 4) Zeichnen
    out  = img.copy()
    orig = CLASS_COLORS.copy()

    # GT = grün
    for k in CLASS_COLORS: CLASS_COLORS[k] = (0,255,0)
    for box,_,_ in gt_vis:
        out = draw_boxes_on_image(out, [box], K, H, rcfg["line_thickness"])

    # Fusion = rot
    CLASS_COLORS.clear(); CLASS_COLORS.update(orig)
    for k in CLASS_COLORS: CLASS_COLORS[k] = (0,0,255)
    for box,_,_ in fus_vis:
        out = draw_boxes_on_image(out, [box], K, H, rcfg["line_thickness"])

    # Matching-Linien
    if gt_vis and fus_vis:
        cost = np.linalg.norm(
            np.array([c for _,_,c in gt_vis])[:,None,:] -
            np.array([c for _,_,c in fus_vis])[None,:,:],
            axis=-1
        )
        row,col = linear_sum_assignment(cost)
        for i,j in zip(row,col):
            p1 = view_points(np.array(gt_vis[i][2]).reshape(3,1), K, normalize=True)[:2,0]
            p2 = view_points(np.array(fus_vis[j][2]).reshape(3,1), K, normalize=True)[:2,0]
            cv2.line(out,
                     (int(p1[0]),int(p1[1])),
                     (int(p2[0]),int(p2[1])),
                     (255,255,0),
                     rcfg["line_thickness"],
                     cv2.LINE_AA)

    os.makedirs(ocfg["fusion_comparison_dir"], exist_ok=True)
    out_fn = os.path.join(
        ocfg["fusion_comparison_dir"],
        f"{sample_i:03d}_{token}_fusion_vs_gt.jpg"
    )
    cv2.imwrite(out_fn, out)
    print(f"✓ Fusion-Vergleich: {out_fn}")

if __name__=="__main__":
    main()