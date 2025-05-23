#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
from pyquaternion import Quaternion
from truckscenes.utils.geometry_utils import transform_matrix

from oft.utils.motion_utils import (
    box_centers_sensor_to_world,
    predict_world_centers,
    box_centers_world_to_sensor,
)
from oft.utils.config     import load_config
from oft.data.dataset     import TruckScenesDataset
from oft.fusion.nms_3d    import nms_bev_3d

def main():
    p = argparse.ArgumentParser(description="Stage 4: Fusion")
    p.add_argument("-c","--pipeline", required=True, help="Pfad zur pipeline.yaml")
    args = p.parse_args()

    # --- Config laden --------------------------------
    cfg    = load_config(args.pipeline)
    dcfg   = cfg["dataset"]
    vcfg   = cfg["visualization"]
    ocfg   = cfg["output"]
    fusion = cfg["fusion"]

    # --- Dataset mit Noise (je nach Pipeline) --------
    common = dict(
        dataroot       = str(dcfg["dataroot"]),
        version        = str(dcfg["version"]).strip(),
        history_window = int(vcfg.get("history_window", 0)),
        max_boxes      = int(dcfg.get("gt_max_boxes") or 0)
    )
    ds = TruckScenesDataset(**common,
                             augment_noise_std=float(dcfg.get("augment_noise_std",0.0)))

    cam_ch    = vcfg["camera_channel"]
    start_idx = int(vcfg.get("sample_idx", 0))
    nframes   = int(vcfg.get("num_frames", 1))
    iou_th    = float(fusion.get("iou_threshold", 0.3))

    entries = []
    # --- Für jedes zu rendernde Sample ----------------
    for frame_idx in range(start_idx, start_idx + nframes):
        item_curr   = ds[frame_idx]
        curr_token  = item_curr["sample_token"]
        ts_curr     = item_curr["timestamp"]

        # 1) Transforms für den aktuellen Frame
        calib_curr = item_curr["calibrated_sensor"][cam_ch]
        q_s_curr   = Quaternion(calib_curr["rotation"])
        E_curr     = transform_matrix(calib_curr["translation"],
                                     q_s_curr,
                                     inverse=False)  # sensor→ego
        pose_curr  = item_curr["ego_pose"][cam_ch]
        q_e_curr   = Quaternion(pose_curr["rotation"])
        T_ego_curr = transform_matrix(pose_curr["translation"],
                                      q_e_curr,
                                      inverse=False)  # ego→world

        # 2) Projiziere alle History‐Frames in den aktuellen Sensor‐Frame
        all_proj = []
        H = common["history_window"]
        for h in range(1, H+1):
            prev_idx = frame_idx - h
            if prev_idx < 0:
                break
            item_prev = ds[prev_idx]
            boxes_prev = item_prev["current"]  # (N,7)
            if boxes_prev.size == 0:
                continue

            ts_prev   = item_prev["timestamp"]
            dt        = (ts_curr - ts_prev) * 1e-6  # in Sekunden

            # 2a) Transforms für Prev‐Frame
            calib_prev = item_prev["calibrated_sensor"][cam_ch]
            q_s_prev   = Quaternion(calib_prev["rotation"])
            E_prev     = transform_matrix(calib_prev["translation"],
                                          q_s_prev,
                                          inverse=False)
            pose_prev  = item_prev["ego_pose"][cam_ch]
            q_e_prev   = Quaternion(pose_prev["rotation"])
            T_ego_prev = transform_matrix(pose_prev["translation"],
                                          q_e_prev,
                                          inverse=False)

            # 2b) Sensor→Welt
            centers_prev = boxes_prev[:, :3]  # x,y,z
            world_prev   = box_centers_sensor_to_world(
                               centers_prev, E_prev, T_ego_prev)

            # 2c) Bewegungs‐Prognose
            vel_prev = item_prev.get("velocities", None)
            if vel_prev is not None:
                world_prev = predict_world_centers(world_prev, vel_prev, dt)

            # 2d) Welt→Sensor (aktueller Frame)
            sensor_pred = box_centers_world_to_sensor(
                              world_prev, E_curr, T_ego_curr)

            # 2e) Rekonstruiere volle 7-Daten (w,l,h,yaw aus Prev)
            dims_yaw = boxes_prev[:, 3:7]
            proj     = np.hstack([sensor_pred, dims_yaw])  # (Ni,7)
            all_proj.append(proj)

        if not all_proj:
            continue

        # 3) NMS über alle projizierten Boxen
        proj_boxes = np.vstack(all_proj)  # (ΣNi,7)
        keep       = nms_bev_3d(proj_boxes, iou_threshold=iou_th)
        fused      = proj_boxes[keep]

        # 4) In JSON‐Format packen
        for b in fused:
            entries.append({
                "sample_token": curr_token,
                "translation": [float(b[0]), float(b[1]), float(b[2])],
                "size":        [float(b[3]), float(b[4]), float(b[5])],
                "rotation":    float(b[6])
            })

    # --- Schreibe JSON ab ----------------------------
    os.makedirs(os.path.dirname(ocfg["dets_json"]), exist_ok=True)
    with open(ocfg["dets_json"], "w") as f:
        json.dump(entries, f, indent=2)

    print(f"✓ Stage 4: wrote {len(entries)} entries → {ocfg['dets_json']}")

if __name__ == "__main__":
    main()