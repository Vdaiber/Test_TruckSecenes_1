#!/usr/bin/env python3
import argparse
import json
import numpy as np
from pyquaternion import Quaternion
from truckscenes.utils.geometry_utils import transform_matrix

from oft.utils.config import load_config
from oft.data.dataset import TruckScenesDataset
from oft.tracking.temporal_tracking import temporal_hungarian_match

def _to_json_serializable(obj):
    """
    Konvertiert numpy-Typen in native Python-Typen für JSON-Dump.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

def main():
    p = argparse.ArgumentParser(description="Stage 3: Track History")
    p.add_argument("-c", "--pipeline", required=True, help="Pfad zur pipeline.yaml")
    args = p.parse_args()

    # 1) Config laden
    cfg  = load_config(args.pipeline)
    dcfg = cfg["dataset"]
    vcfg = cfg["visualization"]
    tc   = cfg.get("tracking", {})

    # 2) Dataset initialisieren (ohne Noise)
    ds = TruckScenesDataset(
        dataroot         = str(dcfg["dataroot"]),
        version          = str(dcfg["version"]).strip(),
        history_window   = int(vcfg.get("history_window", 0)),
        max_boxes        = int(dcfg.get("gt_max_boxes") or 0),
        augment_noise_std= 0.0
    )

    tracks = []
    prev_timestamp = None

    # 3) Tracking über Samples
    for idx in range(len(ds)):
        item = ds[idx]

        # 3.1) dt berechnen (aus Timestamp in µs → s)
        if prev_timestamp is None:
            prev_timestamp = item["timestamp"]
            continue
        curr_timestamp = item["timestamp"]
        dt = (curr_timestamp - prev_timestamp) * 1e-6
        prev_timestamp = curr_timestamp

        # 3.2) Sensor‐Koordinaten der aktuellen und vorherigen Boxes
        curr_sensor = item["current_sensor"]      # (N_curr,7)
        hist_sensor = item["history_sensor"][0]   # (N_prev,7) oder None
        if hist_sensor is None or hist_sensor.size == 0 or curr_sensor.size == 0:
            continue

        prev_xy  = hist_sensor[:, :2]
        curr_xy  = curr_sensor[:, :2]
        prev_xyz = hist_sensor[:, :3]
        prev_vel = item.get("velocities_sensor", None)

        # 3.3) Extrinsik und Ego‐Pose‐Matrizen bauen
        ch = vcfg["camera_channel"]
        ego_prev_dict = item["ego_pose"][ch]
        ego_curr_dict = item["ego_pose"][ch]
        calib         = item["calibrated_sensor"][ch]

        # Ego-Pose (Welt←Ego) Matrizen 4×4
        q_prev = Quaternion(ego_prev_dict["rotation"])
        T_ego_prev = transform_matrix(ego_prev_dict["translation"], q_prev, inverse=False)
        q_curr = Quaternion(ego_curr_dict["rotation"])
        T_ego_curr = transform_matrix(ego_curr_dict["translation"], q_curr, inverse=False)

        # Sensor→Ego‐Extrinsik (Ego←Sensor) 4×4
        q_s = Quaternion(calib["rotation"])
        E = transform_matrix(calib["translation"], q_s, inverse=True)

        # 3.4) Hungarian Matching
        max_distance = float(tc.get("temporal", {}).get("max_distance", 1.0))
        matches = temporal_hungarian_match(
            prev_centers_xy   = prev_xy,
            curr_centers_xy   = curr_xy,
            prev_centers_3d   = prev_xyz,
            prev_velocities   = prev_vel,
            ego_pose_prev     = T_ego_prev,
            ego_pose_curr     = T_ego_curr,
            sensor_extrinsic  = E,
            dt                = dt,
            max_distance      = max_distance
        )

        # 3.5) Matches sammeln
        for gt_idx, pred_idx in matches:
            tracks.append({
                "frame_idx": idx,
                "gt_idx":    gt_idx,
                "pred_idx":  pred_idx
            })

        # nur vcfg["num_frames"] Frames testen
        if idx + 1 >= int(vcfg.get("num_frames", 1)):
            break

    # 4) Ergebnis schreiben
    out_path = cfg["output"]["tracks_json"]
    with open(out_path, "w") as f:
        json.dump(tracks, f, indent=2, default=_to_json_serializable)

    print(f"✓ Stage 3: wrote {len(tracks)} tracks → {out_path}")

if __name__ == "__main__":
    main()