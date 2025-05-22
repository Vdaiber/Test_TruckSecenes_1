#!/usr/bin/env python3
"""
Stage 3: Track History mit Temporal Hungarian Matching
"""
import os
import sys
import yaml
import numpy as np

from truckscenes import TruckScenes
from truckscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

from oft.utils.config import load_config
from oft.utils.sensor_utils import get_camera_intrinsic
from oft.tracking.temporal_tracking import temporal_hungarian_match
from oft.data.dataset import TruckScenesDataset

def main(pipeline_cfg: str = "config/pipeline.yaml"):
    # 1) Config laden
    cfg  = load_config(pipeline_cfg)
    dcfg = cfg["dataset"]
    vcfg = cfg["visualization"]
    tcfg = cfg["tracking"]

    # 2) Dataset und TruckScenes-Objekte
    ds = TruckScenesDataset(
        dataroot       = str(dcfg["dataroot"]),
        version        = str(dcfg["version"]).strip(),
        history_window = int(vcfg.get("history_window", 0)),
        max_boxes      = int(dcfg.get("gt_max_boxes") or 0),
        augment_noise_std = float(dcfg.get("augment_noise_std", 0.0))
    )
    ts = TruckScenes(version=dcfg["version"].strip(), dataroot=dcfg["dataroot"])

    # 3) Parameter
    dt       = float(tcfg["temporal"]["dt"])
    max_dist = float(tcfg["temporal"]["max_distance"])
    cam_ch   = vcfg["camera_channel"]

    print(f"=== Stage 3: {len(ds)} Samples, history_window={ds.history_window} ===")

    # 4) Über alle Paare idx-1 → idx iterieren
    for idx in range(1, len(ds)):
        prev_item = ds[idx-1]
        curr_item = ds[idx]

        prev_arr = prev_item["current"]
        curr_arr = curr_item["current"]
        print(f"[Frame {idx}] prev={prev_arr.shape[0]} Boxes → curr={curr_arr.shape[0]} Boxes")

        if prev_arr.shape[0] == 0 or curr_arr.shape[0] == 0:
            continue

        # 5) Center XY + 3D
        prev_xy = prev_arr[:, :2]
        curr_xy = curr_arr[:, :2]
        prev_xyz = prev_arr[:, :3]

        # 6) Objekt-Geschwindigkeiten (kann None sein)
        prev_vels = curr_item.get("velocities", None)

        # 7) Ego-Pose und Sensor-Extrinsik für prev und curr
        # --- prev ---
        tok_prev = ds.samples[idx-1]
        sd_prev_tk = ts.get("sample", tok_prev)["data"][cam_ch]
        sd_prev    = ts.get("sample_data", sd_prev_tk)
        ego_prev   = ts.get("ego_pose", sd_prev["ego_pose_token"])
        calib_prev = ts.get("calibrated_sensor", sd_prev["calibrated_sensor_token"])
        # world ← ego
        T_prev = transform_matrix(
            translation=ego_prev["translation"],
            rotation=Quaternion(ego_prev["rotation"]),
            inverse=False
        )
        # ego ← sensor
        E       = transform_matrix(
            translation=calib_prev["translation"],
            rotation=Quaternion(calib_prev["rotation"]),
            inverse=True
        )

        # --- curr ---
        tok_curr = ds.samples[idx]
        sd_curr_tk = ts.get("sample", tok_curr)["data"][cam_ch]
        sd_curr    = ts.get("sample_data", sd_curr_tk)
        ego_curr   = ts.get("ego_pose", sd_curr["ego_pose_token"])
        # wir brauchen T_curr nur für world→sensor rücktransform
        T_curr = transform_matrix(
            translation=ego_curr["translation"],
            rotation=Quaternion(ego_curr["rotation"]),
            inverse=False
        )

        # 8) Hungarian Matching
        matches = temporal_hungarian_match(
            prev_centers_xy     = prev_xy,
            curr_centers_xy     = curr_xy,
            prev_centers_3d     = prev_xyz,
            prev_velocities     = prev_vels,
            ego_pose_prev       = T_prev,
            ego_pose_curr       = T_curr,
            sensor_extrinsic    = E,
            dt                  = dt,
            max_distance        = max_dist
        )
        print(f"  → {len(matches)} Matches (prev→curr)")

    print("\n✓ Stage 3 fertig.")

if __name__=="__main__":
    # erlaube -c/--pipeline
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-c","--config", default="config/pipeline.yaml")
    args = p.parse_args()
    main(args.config)