#!/usr/bin/env python3
import os, json, cv2, numpy as np, math
from oft.utils.config import load_config
from oft.utils.sensor_utils import (
    get_camera_intrinsic, get_sensor_extrinsic,
    draw_boxes_on_image, CLASS_COLORS
)
from oft.data.dataset import TruckScenesDataset
from truckscenes import TruckScenes
from truckscenes.utils.data_classes import Box as DevBox

def try_load_image(path):
    img = cv2.imread(path)
    if img is not None:
        return img
    from PIL import Image
    pil = Image.open(path)
    arr = np.array(pil)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr

def main():
    cfg  = load_config()
    dcfg = cfg["dataset"]
    vcfg = cfg["visualization"]
    ocfg = cfg["output"]

    sample_idx = vcfg.get("sample_idx", 0)
    cam_ch     = vcfg["camera_channel"]

    # über Dataset token holen
    ds = TruckScenesDataset(
        dataroot       = dcfg["dataroot"],
        version        = dcfg["version"].strip(),
        history_window = 0,
        max_boxes      = dcfg.get("gt_max_boxes") or 0,
        augment_noise_std = 0.0
    )
    assert 0 <= sample_idx < len(ds), "sample_idx out of range"
    sample_token = ds.samples[sample_idx]

    # Bild + GT über DevKit laden
    ts = TruckScenes(version=dcfg["version"].strip(), dataroot=dcfg["dataroot"])
    sample = ts.get("sample", sample_token)
    sd     = ts.get("sample_data", sample["data"][cam_ch])
    fn     = sd["filename"]
    if not os.path.isabs(fn):
        fn = os.path.join(dcfg["dataroot"], fn)
    img = try_load_image(fn)

    # Matrizen
    calib = ts.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego   = ts.get("ego_pose",         sd["ego_pose_token"])
    K     = get_camera_intrinsic(calib)
    H     = get_sensor_extrinsic(ego, calib)

    # GT-Boxen
    ann_tokens = sample["anns"]
    gt_boxes   = [ ts.get_box(a) for a in ann_tokens ]

    # Fusion-Detektionen
    entries = json.load(open(ocfg["dets_json"], "r"))
    raw     = next((e["dets"].get(cam_ch, [])
                    for e in entries if e["sample_token"] == sample_token), [])
    fus_boxes = []
    for d in raw:
        b = DevBox(
            name        = "fused",
            translation = d["translation"],
            wlh         = d["wlh"],
            rotation    = [0,0,0,1]
        )
        # Quaternion aus yaw
        qx, qy = 0.0, 0.0
        qz = math.sin(d["yaw"]/2)
        qw = math.cos(d["yaw"]/2)
        b.rotation = [qx, qy, qz, qw]
        fus_boxes.append(b)

    # Zeichnen
    orig = CLASS_COLORS.copy()
    # GT = grün
    for k in CLASS_COLORS: CLASS_COLORS[k] = (0,255,0)
    out = draw_boxes_on_image(img, gt_boxes, K, H, cfg["render"]["line_thickness"])
    # Fusion = rot
    CLASS_COLORS.clear(); CLASS_COLORS.update(orig)
    for k in CLASS_COLORS: CLASS_COLORS[k] = (0,0,255)
    out = draw_boxes_on_image(out, fus_boxes, K, H, cfg["render"]["line_thickness"])

    os.makedirs(ocfg["fusion_comparison_dir"], exist_ok=True)
    fn = os.path.join(ocfg["fusion_comparison_dir"],
                      f"{sample_idx:03d}_{sample_token}_fusion_vs_gt.jpg")
    cv2.imwrite(fn, out)
    print(f"✓ Stage 5: wrote {fn}")

if __name__=="__main__":
    main()