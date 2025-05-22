#!/usr/bin/env python3
import os
from oft.utils.config import load_config
from oft.data.dataset import TruckScenesDataset
from oft.tracking.temporal_tracking import temporal_hungarian_match
from oft.utils.sensor_utils import get_sensor_extrinsic
from truckscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

def main():
    cfg  = load_config()
    dcfg = cfg["dataset"]
    vcfg = cfg["visualization"]
    tcfg = cfg["tracking"]

    # Dataset instanziieren (ohne Noise)
    common = dict(
        dataroot        = dcfg["dataroot"],
        version         = dcfg["version"].strip(),
        history_window  = vcfg.get("history_window", 0),
        max_boxes       = dcfg.get("gt_max_boxes") or 0
    )
    ds = TruckScenesDataset(**common, augment_noise_std=0.0)

    cam_ch   = vcfg["camera_channel"]
    dt       = tcfg["temporal"]["dt"]
    max_dist = tcfg["temporal"]["max_distance"]

    print(f"=== Stage 3: Track History ({len(ds)} Samples, history_window={common['history_window']}) ===")
    for idx in range(len(ds)):
        item = ds[idx]
        curr = item["current"]         # (N,7)
        print(f"[Frame {idx}] {curr.shape[0]} GT-Boxen")

        # ab Frame 1 Matching gegen vorherigen
        if idx == 0 or curr.shape[0] == 0:
            continue

        prev_item = ds[idx-1]
        prev = prev_item["current"]
        if prev.shape[0] == 0:
            continue

        # XY für Matching
        prev_xy = prev[:, :2]
        curr_xy = curr[:, :2]
        prev_c3 = prev[:, :3]
        prev_vels = item.get("velocities", None)

        # Ego-Pose als 4×4-Matrix
        ego_prev = prev_item["ego_pose"][cam_ch]
        ego_curr = item["ego_pose"][cam_ch]
        T_prev = transform_matrix(ego_prev["translation"],
                                 Quaternion(ego_prev["rotation"]),
                                 inverse=False)
        T_curr = transform_matrix(ego_curr["translation"],
                                 Quaternion(ego_curr["rotation"]),
                                 inverse=False)

        # Sensor-Extrinsik (Welt→Kamera) im aktuellen Frame
        calib_curr = item["calibrated_sensor"][cam_ch]
        H = get_sensor_extrinsic(ego_curr, calib_curr)

        # Hungarian Matching
        matches = temporal_hungarian_match(
            prev_centers_xy   = prev_xy,
            curr_centers_xy   = curr_xy,
            prev_centers_3d   = prev_c3,
            prev_velocities   = prev_vels,
            ego_pose_prev     = T_prev,
            ego_pose_curr     = T_curr,
            sensor_extrinsic  = H,
            dt                = dt,
            max_distance      = max_dist
        )
        print(f"  → {len(matches)} Matches: {matches}")

if __name__=="__main__":
    main()