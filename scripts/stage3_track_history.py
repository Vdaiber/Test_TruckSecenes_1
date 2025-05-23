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
from oft.utils.common_utils import _to_json_serializable # Import für json.dump

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
        history_window   = int(vcfg.get("history_window", 0)),
        max_boxes        = int(dcfg.get("gt_max_boxes") or 50),
        augment_noise_std= noise_to_apply_for_tracking
    )

    target_sample_idx = int(vcfg.get("sample_idx", 0))
    if target_sample_idx >= len(ds):
        print(f"FEHLER: visualization.sample_idx ({target_sample_idx}) ist außerhalb des Dataset-Bereichs ({len(ds)} Samples).")
        return
    
    target_sample_token = ds.samples[target_sample_idx]
    print(f"INFO Stage 3: Ziel ist es, Tracks für Sample-Index {target_sample_idx} (Token: {target_sample_token}) zu generieren.")

    mot_max_age = int(tc.get("temporal", {}).get("max_age", 5))
    mot_min_hits = int(tc.get("min_hits_to_report", 3))
    mot_match_distance = float(tc.get("temporal", {}).get("max_distance", 10.0))

    mot_tracker = MultiObjectTracker(
        max_age=mot_max_age,
        min_hits_for_output=mot_min_hits,
        match_max_distance=mot_match_distance
    )

    output_tracks_for_target_sample = []
    prev_frame_timestamp = None
    
    # Schleife bis zum target_sample_idx + 1 (um dt für den target_sample_idx zu haben)
    # oder mindestens so viele Frames, dass min_hits erreicht werden kann.
    # Wenn target_sample_idx = 3 und min_hits = 3, müssen wir mind. idx 0,1,2,3 verarbeiten.
    # Die Schleife muss bis `target_sample_idx` laufen, um `mot_tracker.update` für diesen Index aufzurufen.
    # Wir müssen also bis `target_sample_idx + 1` iterieren in der range.
    
    # Wir müssen mindestens `mot_min_hits` Updates durchführen, um überhaupt Tracks zu bekommen.
    # Die Anzahl der Schleifendurchläufe muss also mindestens target_sample_idx + 1 sein,
    # und auch mindestens mot_min_hits (wenn target_sample_idx < mot_min_hits -1).
    # Beispiel: target_idx=0, min_hits=3 -> wir brauchen mind. 3 Updates (Frames 0,1,2,3)
    # Beispiel: target_idx=5, min_hits=3 -> wir brauchen mind. 6 Updates (Frames 0..5)
    
    # Die Anzahl der Iterationen, die wir *mindestens* brauchen, um Tracks für target_sample_idx
    # mit min_hits zu bekommen, ist target_sample_idx + 1 (wenn target_sample_idx >= min_hits - 1).
    # Wenn target_sample_idx < min_hits - 1, brauchen wir min_hits Iterationen.
    # z.B. target_idx=0, min_hits=3 -> wir brauchen 3 Tracker-Updates, also 4 Frames (0,1,2,3).
    # Die Schleife geht von 0 bis `loop_until_idx`.
    # Ein Tracker-Update findet statt für `current_idx > 0`.
    
    loop_until_idx = target_sample_idx
    
    print(f"INFO Stage 3: Verarbeite Frames von Index 0 bis {loop_until_idx} um Tracks für Index {target_sample_idx} zu erhalten.")

    for current_idx in range(loop_until_idx + 1): # +1, damit target_sample_idx als current_idx erreicht wird
        if current_idx >= len(ds):
            print(f"INFO Stage 3: Ende des Datasets bei Index {current_idx-1} erreicht, bevor Ziel-Index {target_sample_idx} verarbeitet werden konnte.")
            break

        item_curr = ds[current_idx]
        curr_timestamp = item_curr["timestamp"]

        if current_idx == 0:
            prev_frame_timestamp = curr_timestamp
            # print(f"INFO Stage 3: Initialisiere prev_timestamp={prev_frame_timestamp} mit Frame idx={current_idx} (Sample: {item_curr['sample_token']})")
            continue

        dt = (curr_timestamp - prev_frame_timestamp) * 1e-6
        # print(f"\n--- Stage 3: Verarbeite Frame idx={current_idx} (Sample: {item_curr['sample_token']}), dt={dt:.4f}s ---")

        if dt <= 1e-9:
            # print(f"WARNUNG Stage 3: dt ({dt:.4f}s) ist zu klein für idx={current_idx}. Überspringe Update.")
            prev_frame_timestamp = curr_timestamp
            continue

        current_boxes_world = item_curr["current"]
        current_velocities_world = item_curr.get("velocities")

        if current_boxes_world.size == 0:
            # print(f"INFO Stage 3: Keine Detektionen im aktuellen Frame idx={current_idx}.")
            current_boxes_world = np.zeros((0,7), dtype=np.float32)
            if current_velocities_world is not None:
                 current_velocities_world = np.zeros((0,3), dtype=np.float32)
        
        if current_velocities_world is not None and current_velocities_world.shape[0] != current_boxes_world.shape[0]:
            # print(f"WARNUNG Stage 3: Vel-Shape-Mismatch für idx={current_idx}. Setze velocities auf None.")
            current_velocities_world = None

        active_tracks = mot_tracker.update(
            current_detections_world=current_boxes_world,
            current_velocities_world=current_velocities_world,
            dt=dt
        )
        
        # Speichere nur die Tracks für den Ziel-Sample-Token
        if item_curr["sample_token"] == target_sample_token:
            print(f"  => MOT ergab {len(active_tracks)} aktive Tracks für ZIEL-Frame idx={current_idx} (Token: {target_sample_token}).")
            if active_tracks: # Nur wenn es Tracks gibt
                output_tracks_for_target_sample = [{ # Speichere als Liste mit einem Element für Konsistenz mit altem Format
                    "sample_token": item_curr["sample_token"],
                    "timestamp": item_curr["timestamp"],
                    "dt_to_prev_frame": dt,
                    "num_input_detections": current_boxes_world.shape[0],
                    "tracks": active_tracks
                }]
            else: # Auch wenn keine Tracks, aber es ist der Ziel-Frame
                 output_tracks_for_target_sample = [{
                    "sample_token": item_curr["sample_token"],
                    "timestamp": item_curr["timestamp"],
                    "dt_to_prev_frame": dt,
                    "num_input_detections": current_boxes_world.shape[0],
                    "tracks": []
                }]


        prev_frame_timestamp = curr_timestamp
            
    out_path = cfg["output"]["tracks_json"]
    output_dir_for_json = os.path.dirname(out_path)
    if not os.path.exists(output_dir_for_json) and output_dir_for_json:
        os.makedirs(output_dir_for_json, exist_ok=True)
    
    with open(out_path, "w") as f:
        # Speichere nur die Tracks des Ziel-Frames (oder eine leere Liste, wenn keine gefunden wurden)
        json.dump(output_tracks_for_target_sample, f, indent=2, default=_to_json_serializable)

    num_total_tracks_in_file = sum(len(frame_data.get("tracks", [])) for frame_data in output_tracks_for_target_sample)
    print(f"✓ Stage 3: wrote {num_total_tracks_in_file} tracks für Sample {target_sample_token} → {out_path}")

if __name__ == "__main__":
    main()