#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
# PyQuaternion und transform_matrix werden hier nicht mehr direkt benötigt
# from pyquaternion import Quaternion
# from truckscenes.utils.geometry_utils import transform_matrix

from oft.utils.config import load_config
from oft.data.dataset import TruckScenesDataset
from oft.utils.tracking_utils import MultiObjectTracker 
from oft.utils.common_utils import _to_json_serializable

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Track History for a target sample_idx")
    parser.add_argument("-c", "--pipeline", required=False, default="config/pipeline.yaml",
                        help="Pfad zur pipeline.yaml. Standard: config/pipeline.yaml im CWD.")
    args = parser.parse_args()

    cfg  = load_config(args.pipeline)
    dcfg = cfg["dataset"]
    vcfg = cfg["visualization"]
    tc   = cfg.get("tracking", {})

    noise_to_apply_for_tracking = 0.0
    if tc.get("use_noisy_gt_as_input", False):
        noise_to_apply_for_tracking = float(dcfg.get("augment_noise_std", 0.0))
        print(f"INFO Stage 3: Dataset wird mit augment_noise_std={noise_to_apply_for_tracking} initialisiert (verrauschte GTs).")
    else:
        print(f"INFO Stage 3: Dataset wird ohne zusätzliches Rauschen initialisiert (saubere GTs).")

    ds = TruckScenesDataset(
        dataroot         = str(dcfg["dataroot"]),
        version          = str(dcfg["version"]).strip(),
        history_window   = int(vcfg.get("history_window", 3)), 
        max_boxes        = int(dcfg.get("gt_max_boxes") or 50),
        augment_noise_std= noise_to_apply_for_tracking
    )

    target_sample_idx = int(vcfg.get("sample_idx", 0))
    if target_sample_idx >= len(ds):
        print(f"FEHLER: visualization.sample_idx ({target_sample_idx}) ist außerhalb des Dataset-Bereichs ({len(ds)} Samples).")
        return
    
    target_sample_token = ds.samples[target_sample_idx]
    print(f"INFO Stage 3: Ziel ist es, Tracks für Sample-Index {target_sample_idx} (Token: {target_sample_token}) zu generieren.")

    mot_max_age = int(tc.get("temporal", {}).get("max_age", 3)) 
    mot_min_hits = int(tc.get("min_hits_to_report", 3))    
    mot_match_distance = float(tc.get("temporal", {}).get("max_distance", 5.0)) 

    mot_tracker = MultiObjectTracker(
        max_age=mot_max_age,
        min_hits_for_output=mot_min_hits,
        match_max_distance=mot_match_distance
    )

    output_tracks_for_target_sample_dict = {} 
    prev_frame_timestamp = None
    
    loop_until_this_index_inclusive = target_sample_idx
    
    print(f"INFO Stage 3: Verarbeite Frames von Index 0 bis {loop_until_this_index_inclusive} um Tracks für Ziel-Index {target_sample_idx} zu erhalten.")

    for current_idx in range(loop_until_this_index_inclusive + 1): 
        if current_idx >= len(ds):
            print(f"INFO Stage 3: Ende des Datasets bei Index {current_idx-1} erreicht, bevor Ziel-Index {target_sample_idx} verarbeitet werden konnte.")
            break

        item_curr = ds[current_idx]
        curr_timestamp = item_curr["timestamp"]
        
        if current_idx == 0: 
            prev_frame_timestamp = curr_timestamp
            continue

        dt = (curr_timestamp - prev_frame_timestamp) * 1e-6
        
        if dt <= 1e-9: 
            prev_frame_timestamp = curr_timestamp 
            continue

        current_boxes_world = item_curr["current"]
        current_velocities_world = item_curr.get("velocities")

        if current_boxes_world.size == 0:
            current_boxes_world = np.zeros((0,7), dtype=np.float32)
            if current_velocities_world is not None:
                 current_velocities_world = np.zeros((0,3), dtype=np.float32)
        
        if current_velocities_world is not None and current_velocities_world.shape[0] != current_boxes_world.shape[0]:
            current_velocities_world = None 

        active_tracks = mot_tracker.update(
            current_detections_world=current_boxes_world,
            current_velocities_world=current_velocities_world,
            dt=dt
        )
        
        if current_idx == target_sample_idx:
            print(f"  => MOT ergab {len(active_tracks)} aktive Tracks für ZIEL-Frame idx={current_idx} (Token: {item_curr['sample_token']}).")
            output_tracks_for_target_sample_dict = {
                "sample_token": item_curr["sample_token"],
                "timestamp": item_curr["timestamp"],
                "dt_to_prev_frame": dt,
                "num_input_detections_in_target_frame": current_boxes_world.shape[0],
                "tracks": active_tracks
            }

        prev_frame_timestamp = curr_timestamp
            
    out_path = cfg["output"]["tracks_json"]
    output_dir_for_json = os.path.dirname(out_path)
    if not os.path.exists(output_dir_for_json) and output_dir_for_json:
        os.makedirs(output_dir_for_json, exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(output_tracks_for_target_sample_dict, f, indent=2, default=_to_json_serializable)

    num_tracks_in_file = len(output_tracks_for_target_sample_dict.get("tracks", []))
    if output_tracks_for_target_sample_dict:
        print(f"✓ Stage 3: wrote {num_tracks_in_file} tracks für Sample {output_tracks_for_target_sample_dict.get('sample_token')} → {out_path}")
    else:
        print(f"WARNUNG Stage 3: Keine Tracks für Ziel-Sample-Index {target_sample_idx} (Token {target_sample_token}) geschrieben.")

if __name__ == "__main__":
    main()