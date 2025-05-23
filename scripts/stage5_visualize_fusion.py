#!/usr/bin/env python3
import os, json, cv2, math, numpy as np
from pyquaternion import Quaternion
from oft.utils.config import load_config
from oft.data.dataset import TruckScenesDataset
from truckscenes import TruckScenes
from truckscenes.utils.data_classes import Box as DevBox
from oft.utils.sensor_utils import (
    get_camera_intrinsic, get_sensor_extrinsic, draw_boxes_on_image, CLASS_COLORS
)

def try_load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    from PIL import Image
    arr = np.array(Image.open(path))
    if arr.ndim==2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def main():
    cfg   = load_config()
    dcfg  = cfg["dataset"]
    vcfg  = cfg["visualization"]
    ocfg  = cfg["output"]
    cam_ch = vcfg["camera_channel"]

    # Welches Sample zeichnen?
    i = vcfg.get("sample_idx", 0)
    ds = TruckScenesDataset(
        dataroot       = dcfg["dataroot"],
        version        = dcfg["version"].strip(),
        history_window = 0,
        max_boxes      = dcfg.get("gt_max_boxes", 0),
        augment_noise_std = 0.0
    )
    token = ds.samples[i]

    # Bild & GT via DevKit
    ts     = TruckScenes(version=dcfg["version"].strip(), dataroot=dcfg["dataroot"])
    samp   = ts.get("sample", token)
    sd     = ts.get("sample_data", samp["data"][cam_ch])
    img_fn = sd["filename"]
    if not os.path.isabs(img_fn):
        img_fn = os.path.join(dcfg["dataroot"], img_fn)
    img    = try_load_image(img_fn)

    calib = ts.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego   = ts.get("ego_pose",         sd["ego_pose_token"])
    K     = get_camera_intrinsic(calib)
    H     = get_sensor_extrinsic(ego, calib)

    # Ground‐Truth
    gt_boxes = [ ts.get_box(a) for a in samp["anns"] ]

    # History-Fusion-Detektionen
    fused = json.load(open(ocfg["fused_json"], "r"))
    raw   = next(e for e in fused if e["sample_token"]==token)["dets"].get(cam_ch, [])

    fus_boxes = []
    for d in raw:
        b = DevBox(
            name        = "fused",
            translation = d["translation"],
            wlh         = d["wlh"],
            rotation    = [0,0,0,1]
        )
        # echte Quaternion aus yaw
        qx, qy = 0.0, 0.0
        qz = math.sin(d["yaw"]/2)
        qw = math.cos(d["yaw"]/2)
        b.rotation = [qx, qy, qz, qw]
        fus_boxes.append(b)

    # 1) GT in grün
    orig = CLASS_COLORS.copy()
    for k in CLASS_COLORS: CLASS_COLORS[k] = (0,255,0)
    out = draw_boxes_on_image(img, gt_boxes, K, H, cfg["render"]["line_thickness"])

    # 2) Fusion in rot
    CLASS_COLORS.clear(); CLASS_COLORS.update(orig)
    for k in CLASS_COLORS: CLASS_COLORS[k] = (0,0,255)
    out = draw_boxes_on_image(out, fus_boxes, K, H, cfg["render"]["line_thickness"])

    os.makedirs(ocfg["fusion_comparison_dir"], exist_ok=True)
    fn = os.path.join(ocfg["fusion_comparison_dir"],
                      f"{i:03d}_{token}_history_fusion_vs_gt.jpg")
    cv2.imwrite(fn, out)
    print(f"✓ Stage 5: wrote {fn}")

if __name__ == "__main__":
    main()