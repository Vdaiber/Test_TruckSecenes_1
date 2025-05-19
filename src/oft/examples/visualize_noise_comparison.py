#!/usr/bin/env python3
import os
import sys
import yaml
import cv2
import numpy as np
import copy

# ensure repo/src on PYTHONPATH
HERE     = os.path.abspath(__file__)
SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from oft.data.dataset import TruckScenesDataset
from oft.utils.sensor_utils import (
    get_camera_intrinsic,
    get_sensor_extrinsic,
    draw_boxes_on_image,
    CLASS_COLORS,
)
from truckscenes.utils.geometry_utils import view_points


def load_config(path="config/pipeline.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def filter_visible(boxes, K, H, img_shape):
    H_img, W_img = img_shape[:2]
    kept = []
    for box in boxes:
        corners = box.corners()                            # (3,8)
        homo    = np.vstack((corners, np.ones((1,8))))    # (4,8)
        cam_pts = H @ homo                                 # (4,8)
        zs      = cam_pts[2]
        if not np.all(zs > 0):                             # all in front?
            continue
        img_pts = view_points(cam_pts[:3], K, normalize=True)
        xs, ys  = img_pts[0], img_pts[1]
        if np.any((xs>=0)&(xs<W_img)&(ys>=0)&(ys<H_img)):
            kept.append(box)
    return kept


def main():
    cfg = load_config()

    # pull from config
    dcfg      = cfg["dataset"]
    vcfg      = cfg["visualization"]
    rcfg      = cfg["render"]
    out_dir   = cfg["output"]["noise_comparison_dir"]
    cam_ch    = vcfg["camera_channel"]
    sample_i  = vcfg.get("sample_idx", 0)
    thickness = rcfg["line_thickness"]

    os.makedirs(out_dir, exist_ok=True)

    common = dict(
        dataroot       = dcfg["dataroot"],
        version        = dcfg["version"].strip(),
        history_window = vcfg.get("history_window", 0),
        max_boxes      = dcfg.get("gt_max_boxes") or 0
    )

    # two views: clean vs noisy
    ds_clean = TruckScenesDataset(**common, augment_noise_std=0.0)
    ds_noisy = TruckScenesDataset(**common, augment_noise_std=dcfg.get("augment_noise_std",0.0))

    # bounds check
    if not (0 <= sample_i < len(ds_clean)):
        raise IndexError(f"sample_idx {sample_i} out of range")

    # same sample token & sample_data record
    tok    = ds_clean.samples[sample_i]
    samp   = ds_clean.ts.get("sample", tok)
    sd_tok = samp["data"][cam_ch]
    sd     = ds_clean.ts.get("sample_data", sd_tok)
    img_fn = sd["filename"]
    if not os.path.isabs(img_fn):
        img_fn = os.path.join(ds_clean.ts.dataroot, img_fn)

    img = cv2.imread(img_fn)
    if img is None:
        raise FileNotFoundError(f"could not read image {img_fn}")

    # intrinsics/extrinsics
    calib = ds_clean.ts.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego   = ds_clean.ts.get("ego_pose",       sd["ego_pose_token"])
    K     = get_camera_intrinsic(calib)
    H     = get_sensor_extrinsic(ego, calib)

    # ground‐truth boxes
    gt_boxes = [ds_clean.ts.get_box(a) for a in samp["anns"]]

    # noisy copies
    noisy_arrs  = ds_noisy[sample_i]["current"]
    noisy_boxes = []
    for gt, arr in zip(gt_boxes, noisy_arrs):
        nb = copy.copy(gt)
        nb.center = (float(arr[0]), float(arr[1]), float(arr[2]))
        noisy_boxes.append(nb)

    # filter out any that end up off‐screen
    gt_boxes    = filter_visible(gt_boxes,   K, H, img.shape)
    noisy_boxes = filter_visible(noisy_boxes, K, H, img.shape)

    # temporarily override colors
    orig = CLASS_COLORS.copy()

    # 1) draw GT in green
    for k in CLASS_COLORS: CLASS_COLORS[k] = (0,255,0)
    out = draw_boxes_on_image(img, gt_boxes,   K, H, thickness)

    # restore then set red
    CLASS_COLORS.clear(); CLASS_COLORS.update(orig)
    for k in CLASS_COLORS: CLASS_COLORS[k] = (0,0,255)
    out = draw_boxes_on_image(out, noisy_boxes, K, H, thickness)

    # finally restore
    CLASS_COLORS.clear(); CLASS_COLORS.update(orig)

    # write
    fn = os.path.join(out_dir, f"sample_{sample_i:03d}_clean_vs_noisy.jpg")
    cv2.imwrite(fn, out)
    print("wrote", fn)


if __name__=="__main__":
    main() 