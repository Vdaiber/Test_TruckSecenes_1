#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Interne Projekt-Imports
from oft.utils.config import load_config
from oft.data.dataset import TruckScenesDataset # Für Timestamps und Sequenz-Logik
from oft.utils.tracking_utils import MultiObjectTracker # Dein Tracker
from oft.utils.common_utils import _to_json_serializable # Dein JSON-Serialisierungshelfer

def load_detections_from_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray, str, List[str], List[float]]:
    """
    Lädt Detektionen aus einer JSON-Datei, die von A1 erstellt wurde.
    Gibt Boxen (N,7), Geschwindigkeiten (N,3), den Sample-Token, detection_names und scores zurück.
    """
    if not filepath.is_file():
        # print(f"    WARNUNG B1: Detektionsdatei nicht gefunden: {filepath}")
        return np.zeros((0, 7)), np.zeros((0, 3)), "", [], []

    with open(filepath, 'r') as f:
        data = json.load(f)

    sample_token = data.get("meta", {}).get("source_sample_token")
    if not sample_token: 
        if data.get("results"):
            sample_token = list(data["results"].keys())[0] if data["results"] else ""
        else:
            return np.zeros((0, 7)), np.zeros((0, 3)), "", [], []

    detections_list = data.get("results", {}).get(sample_token, [])
    
    boxes_7d_list = []
    velocities_3d_list = []
    detection_names_list = []
    detection_scores_list = []

    for det in detections_list:
        box = det.get("translation_world", [0,0,0]) + \
              det.get("size_wlh", [0,0,0]) + \
              [det.get("rotation_yaw_world", 0.0)]
        boxes_7d_list.append(box)
        velocities_3d_list.append(det.get("velocity_world", [0,0,0]))
        detection_names_list.append(det.get("detection_name", "unknown"))
        detection_scores_list.append(det.get("detection_score", 0.0))

    return (
        np.array(boxes_7d_list, dtype=np.float32), 
        np.array(velocities_3d_list, dtype=np.float32), 
        sample_token,
        detection_names_list,
        detection_scores_list
    )

def main():
    parser = argparse.ArgumentParser(description="B1: Track Simulated Sensor Outputs")
    parser.add_argument("-c", "--pipeline", required=False, default="config/pipeline.yaml",
                        help="Pfad zur pipeline.yaml.")
    args = parser.parse_args()

    cfg = load_config(args.pipeline)
    dcfg = cfg["dataset"]
    vcfg = cfg["visualization"]
    ocfg = cfg["output"]
    sim_cfg = cfg.get("simulation", {})
    tracker_cfg = cfg.get("tracking", {})

    if not sim_cfg.get("enabled", False) or not tracker_cfg.get("enabled", False):
        print("INFO B1: Simulation oder Tracking ist in der pipeline.yaml deaktiviert. Überspringe B1.")
        return

    sequential_sim_base_dir_str = ocfg.get("simulated_detections_dir_A1", "output/simulated_detections_sequence") 
    sequential_sim_base_dir = Path(sequential_sim_base_dir_str)
    
    intermediate_tracks_base_dir_str = ocfg.get("intermediate_tracks_dir_B1", "output/intermediate_tracks") 
    intermediate_tracks_base_dir = Path(intermediate_tracks_base_dir_str)
    intermediate_tracks_base_dir.mkdir(parents=True, exist_ok=True)

    target_sample_idx = int(vcfg.get("sample_idx", 0))
    history_window_for_sequence = int(vcfg.get("history_window", 0)) 
    # DEBUG: Gib die geladenen Werte aus
    print(f"DEBUG B1: target_sample_idx aus Config: {target_sample_idx}")
    print(f"DEBUG B1: history_window_for_sequence aus Config: {history_window_for_sequence}")


    max_age = int(tracker_cfg.get("max_age", 3))
    min_hits_for_output = int(tracker_cfg.get("min_hits_to_report", 3))
    match_max_distance = float(tracker_cfg.get("temporal", {}).get("max_distance", 5.0))

    print(f"INFO B1: Initialisiere TruckScenesDataset für Timestamp-Informationen...")
    ds_helper = TruckScenesDataset(
        dataroot=str(dcfg["dataroot"]),
        version=str(dcfg["version"]).strip(),
        history_window=0, 
        max_boxes=0,      
        augment_noise_std=0.0
    )

    if not (0 <= target_sample_idx < len(ds_helper.samples)):
        print(f"FEHLER B1: visualization.sample_idx ({target_sample_idx}) ist außerhalb des Dataset-Bereichs.")
        return
        
    sample_tokens_in_sequence: List[str] = []
    current_token_for_seq_build = ds_helper.samples[target_sample_idx]
    print(f"DEBUG B1: Start-Token für Sequenzerstellung (target_sample_idx={target_sample_idx}): {current_token_for_seq_build}")
    
    # DEBUG: Wie oft soll die Schleife laufen?
    num_iterations_for_loop = history_window_for_sequence + 1
    print(f"DEBUG B1: Schleife zur Token-Sammlung wird {num_iterations_for_loop} Mal durchlaufen (sollen).")

    for i in range(num_iterations_for_loop): # Geändert zu num_iterations_for_loop für Klarheit
        print(f"DEBUG B1: Token-Sammel-Iteration {i+1}/{num_iterations_for_loop}. Aktueller Token: {current_token_for_seq_build}")
        sample_tokens_in_sequence.append(current_token_for_seq_build)
        sample_record = ds_helper.ts.get("sample", current_token_for_seq_build)
        prev_token = sample_record["prev"]
        print(f"DEBUG B1: prev_token für {current_token_for_seq_build} ist '{prev_token}'")
        if not prev_token: 
            print(f"DEBUG B1: Kein prev_token gefunden. Breche Token-Sammlung ab.")
            break
        current_token_for_seq_build = prev_token
    
    sample_tokens_in_sequence.reverse()
    print(f"DEBUG B1: Gesammelte Tokens (chronologisch): {sample_tokens_in_sequence}")
    print(f"DEBUG B1: Anzahl gesammelter Tokens: {len(sample_tokens_in_sequence)}")


    if not sample_tokens_in_sequence:
        print("FEHLER B1: Keine Sample-Tokens für die Tracking-Sequenz gefunden.")
        return
    
    target_sample_token_for_output = sample_tokens_in_sequence[-1] 
    # Die folgende INFO-Zeile wurde von dir schon als "Verarbeite 1 Frames..." gezeigt,
    # das deutet darauf hin, dass len(sample_tokens_in_sequence) hier 1 ist.
    print(f"INFO B1: Verarbeite {len(sample_tokens_in_sequence)} Frames für jeden Sensor, endend mit Token {target_sample_token_for_output}.")

    simulated_sources_config: Dict[str, Any] = sim_cfg.get("sources", {})
    if not simulated_sources_config:
        print("WARNUNG B1: Keine 'simulation.sources' in der pipeline.yaml definiert. Keine Tracks generiert.")
        return

    for sensor_name in simulated_sources_config.keys():
        print(f"\n  Verarbeite Tracking für Sensor: {sensor_name}")
        
        tracker = MultiObjectTracker(
            max_age=max_age,
            min_hits_for_output=min_hits_for_output,
            match_max_distance=match_max_distance
        )
        
        last_timestamp = None
        output_tracks_for_this_sensor: List[Dict] = []

        for frame_idx, current_frame_token in enumerate(sample_tokens_in_sequence):
            # print(f"    Frame {frame_idx + 1}/{len(sample_tokens_in_sequence)} (Token: {current_frame_token})")
            
            sensor_frame_data_path = sequential_sim_base_dir / sensor_name / f"{current_frame_token}.json"
            boxes_7d, velocities_3d, loaded_token, _detection_names, _detection_scores = load_detections_from_file(sensor_frame_data_path)

            if loaded_token != current_frame_token and loaded_token != "": 
                print(f"      WARNUNG B1: Geladener Token '{loaded_token}' stimmt nicht mit erwartetem Token '{current_frame_token}' überein.")
            
            if boxes_7d.shape[0] == 0:
                pass 

            current_timestamp = ds_helper.ts.get("sample", current_frame_token)["timestamp"]
            dt = 0.05 
            if last_timestamp is not None and current_timestamp > last_timestamp:
                dt = (current_timestamp - last_timestamp) / 1_000_000.0 
            
            if dt <= 0: 
                dt = 0.05 
            
            active_tracks_at_this_frame = tracker.update(boxes_7d, velocities_3d, dt)
            last_timestamp = current_timestamp

            if current_frame_token == target_sample_token_for_output:
                output_tracks_for_this_sensor = active_tracks_at_this_frame
                print(f"    Tracks für ZIEL-FRAME {target_sample_token_for_output} (Sensor: {sensor_name}): {len(output_tracks_for_this_sensor)} Tracks")

        output_filename = f"tracks_{sensor_name}_target_{target_sample_token_for_output}.json"
        output_filepath = intermediate_tracks_base_dir / output_filename
        
        with open(output_filepath, "w") as f:
            json.dump(output_tracks_for_this_sensor, f, indent=2, default=_to_json_serializable)
        print(f"    Finale Tracks für Sensor {sensor_name} (für Token {target_sample_token_for_output}) gespeichert in: {output_filepath}")

    print(f"\n✓ B1: Tracking für alle simulierten Sensoren abgeschlossen.")

if __name__ == "__main__":
    main()