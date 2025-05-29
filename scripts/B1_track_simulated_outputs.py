#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import traceback # Import für Fehler-Stacktrace

# Interne Projekt-Imports
from oft.utils.config import load_config
from oft.data.dataset import TruckScenesDataset 
from oft.utils.tracking_utils import MultiObjectTracker 
from oft.utils.common_utils import _to_json_serializable 

# (load_detections_from_file Funktion bleibt unverändert)
def load_detections_from_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray, str, List[str], List[float]]:
    """
    Lädt Detektionen aus einer JSON-Datei, die von A1 erstellt wurde.
    Gibt Boxen (N,7), Geschwindigkeiten (N,3), den Sample-Token, detection_names und scores zurück.
    """
    if not filepath.is_file():
        print(f"    DEBUG B1 IO: Detektionsdatei NICHT GEFUNDEN: {filepath}") # Hinzugefügt
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), "", [], []

    # print(f"    DEBUG B1 IO: Versuche Detektionsdatei zu laden: {filepath}") # Optional: Sehr gesprächig
    with open(filepath, 'r') as f:
        data = json.load(f)

    sample_token = data.get("meta", {}).get("source_sample_token")
    if not sample_token: 
        if data.get("results"):
            sample_token = list(data["results"].keys())[0] if data["results"] else ""
        else: 
            print(f"WARNUNG B1 IO: Kein Sample-Token in Datei {filepath} gefunden.")
            return np.zeros((0, 7), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), "", [], []

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
    
    # print(f"    DEBUG B1 IO: Datei {filepath} geladen. {len(boxes_7d_list)} Detektionen.") # Optional
    return (
        np.array(boxes_7d_list, dtype=np.float32), 
        np.array(velocities_3d_list, dtype=np.float32), 
        sample_token,
        detection_names_list,
        detection_scores_list
    )

def main():
    print("DEBUG B1: main() gestartet.") # Hinzugefügt
    parser = argparse.ArgumentParser(description="B1: Track Simulated Sensor Outputs")
    parser.add_argument("-c", "--pipeline", required=False, default="config/pipeline.yaml",
                        help="Pfad zur pipeline.yaml.")
    args = parser.parse_args()

    try: # Umfassender try-except Block
        cfg = load_config(args.pipeline)
        dcfg = cfg["dataset"]
        vcfg = cfg["visualization"]
        ocfg = cfg["output"]
        sim_cfg = cfg.get("simulation", {})
        tracker_cfg_from_yaml = cfg.get("tracking", {}) 
        print("DEBUG B1: Konfiguration geladen.") # Hinzugefügt

        if not sim_cfg.get("enabled", False) or not tracker_cfg_from_yaml.get("enabled", False):
            print("INFO B1: Simulation oder Tracking ist in der pipeline.yaml deaktiviert. Überspringe B1.")
            return

        sequential_sim_base_dir_str = ocfg.get("simulated_detections_dir_A1", "output/simulated_detections_sequence") 
        sequential_sim_base_dir = Path(sequential_sim_base_dir_str)
        
        intermediate_tracks_base_dir_str = ocfg.get("intermediate_tracks_dir_B1", "output/intermediate_tracks") 
        intermediate_tracks_base_dir = Path(intermediate_tracks_base_dir_str)
        intermediate_tracks_base_dir.mkdir(parents=True, exist_ok=True)
        print(f"DEBUG B1: Input-Dir (A1): {sequential_sim_base_dir.resolve()}") # Hinzugefügt
        print(f"DEBUG B1: Output-Dir (B1): {intermediate_tracks_base_dir.resolve()}") # Hinzugefügt


        target_sample_idx = int(vcfg.get("sample_idx", 0))
        history_window_for_sequence = int(vcfg.get("history_window", 0)) 
                                             
        print(f"INFO B1: Initialisiere TruckScenesDataset für Timestamp-Informationen...")
        ds_helper = TruckScenesDataset(
            dataroot=str(dcfg["dataroot"]),
            version=str(dcfg["version"]).strip(),
            history_window=0, 
            max_boxes=0,      
            augment_noise_std=0.0 
        )
        print(f"DEBUG B1: ds_helper initialisiert. Anzahl Samples: {len(ds_helper.samples)}") # Hinzugefügt

        if not (0 <= target_sample_idx < len(ds_helper.samples)):
            print(f"FEHLER B1: visualization.sample_idx ({target_sample_idx}) ist außerhalb des Dataset-Bereichs.")
            return
            
        sample_tokens_in_sequence: List[str] = []
        current_token_for_seq_build = ds_helper.samples[target_sample_idx]
        num_frames_in_sequence = history_window_for_sequence + 1

        for i_seq in range(num_frames_in_sequence): 
            print(f"  DEBUG B1 SEQ: Baue Sequenz, Frame {i_seq+1}/{num_frames_in_sequence}, aktueller Token: {current_token_for_seq_build}") # Hinzugefügt
            sample_tokens_in_sequence.append(current_token_for_seq_build)
            sample_record = ds_helper.ts.get("sample", current_token_for_seq_build)
            if not sample_record: # Sollte nicht passieren, wenn ds_helper.samples korrekt ist
                print(f"FEHLER B1 SEQ: Sample Record für Token {current_token_for_seq_build} nicht gefunden!")
                break 
            prev_token = sample_record["prev"]
            if not prev_token: 
                print(f"  DEBUG B1 SEQ: Anfang der Szene für Token {current_token_for_seq_build} erreicht.") # Hinzugefügt
                break 
            current_token_for_seq_build = prev_token
        
        sample_tokens_in_sequence.reverse() 

        if not sample_tokens_in_sequence:
            print("FEHLER B1: Keine Sample-Tokens für die Tracking-Sequenz gefunden.")
            return
        
        target_sample_token_for_output = sample_tokens_in_sequence[-1] 
        print(f"INFO B1: Verarbeite {len(sample_tokens_in_sequence)} Frames für jeden Sensor, endend mit Token {target_sample_token_for_output}.")
        print(f"DEBUG B1: Zu verarbeitende Sequenz-Tokens: {sample_tokens_in_sequence}") # Hinzugefügt

        simulated_sources_config: Dict[str, Any] = sim_cfg.get("sources", {})
        if not simulated_sources_config:
            print("WARNUNG B1: Keine 'simulation.sources' in der pipeline.yaml definiert. Keine Tracks generiert.")
            return

        print(f"DEBUG B1: Beginne Verarbeitung für Sensoren: {list(simulated_sources_config.keys())}") # Hinzugefügt
        for sensor_name in simulated_sources_config.keys():
            print(f"\n  DEBUG B1 SENSOR: Verarbeite Tracking für Sensor: {sensor_name}") # Hinzugefügt
            
            tracker = MultiObjectTracker(tracker_config=tracker_cfg_from_yaml)
            print(f"    DEBUG B1 SENSOR: Tracker für {sensor_name} initialisiert.") # Hinzugefügt
            
            last_timestamp = None
            output_tracks_for_this_sensor: List[Dict] = []

            for frame_idx, current_frame_token in enumerate(sample_tokens_in_sequence):
                print(f"    DEBUG B1 FRAME: Sensor {sensor_name}, Frame {frame_idx + 1}/{len(sample_tokens_in_sequence)} (Token: {current_frame_token})") # Hinzugefügt
                
                sensor_frame_data_path = sequential_sim_base_dir / sensor_name / f"{current_frame_token}.json"
                boxes_7d, velocities_3d, loaded_token, detection_names, detection_scores = load_detections_from_file(sensor_frame_data_path)
                print(f"      DEBUG B1 FRAME: {boxes_7d.shape[0]} Detektionen geladen aus {sensor_frame_data_path}.") # Hinzugefügt

                if loaded_token != current_frame_token and loaded_token != "": 
                    print(f"      WARNUNG B1 FRAME: Geladener Token '{loaded_token}' stimmt nicht mit erwartetem Token '{current_frame_token}' überein.")
                
                if boxes_7d.shape[0] == 0 and velocities_3d.shape[0] != 0:
                    velocities_3d = np.zeros((0,3), dtype=np.float32)
                elif boxes_7d.shape[0] != velocities_3d.shape[0] and velocities_3d.shape[0] !=0 :
                    print(f"      WARNUNG B1 FRAME: Inkonsistente Anzahl Boxen ({boxes_7d.shape[0]}) und Velocities ({velocities_3d.shape[0]}). Setze Velocities auf None.")
                    velocities_3d_for_tracker = None
                else:
                    velocities_3d_for_tracker = velocities_3d
                
                if velocities_3d_for_tracker is not None:
                     print(f"      DEBUG B1 FRAME: Velocities für Tracker (Shape): {velocities_3d_for_tracker.shape if isinstance(velocities_3d_for_tracker, np.ndarray) else 'None'}") # Hinzugefügt
                else:
                     print(f"      DEBUG B1 FRAME: Velocities für Tracker: None") # Hinzugefügt


                current_sample_record_for_ts = ds_helper.ts.get("sample", current_frame_token)
                if not current_sample_record_for_ts:
                     print(f"      FEHLER B1 FRAME: Konnte Sample Record für Timestamp von Token {current_frame_token} nicht laden! Überspringe Frame-Update für Tracker.")
                     continue # Zum nächsten Frame springen
                current_timestamp = current_sample_record_for_ts["timestamp"]

                dt = 0.05 
                if last_timestamp is not None and current_timestamp > last_timestamp:
                    dt = (current_timestamp - last_timestamp) / 1_000_000.0 
                if dt <= 1e-9: dt = 0.05 
                print(f"      DEBUG B1 FRAME: Verwendetes dt für Tracker-Update: {dt:.4f}s") # Hinzugefügt
                
                active_tracks_at_this_frame = tracker.update(boxes_7d, velocities_3d_for_tracker, dt)
                print(f"      DEBUG B1 FRAME: Tracker.update() ergab {len(active_tracks_at_this_frame)} 'aktive' Tracks (die Output-Kriterien erfüllen).") # Hinzugefügt
                last_timestamp = current_timestamp

                if current_frame_token == target_sample_token_for_output:
                    output_tracks_for_this_sensor = active_tracks_at_this_frame
                    print(f"    DEBUG B1 TARGET_FRAME: Tracks für ZIEL-FRAME {target_sample_token_for_output} (Sensor: {sensor_name}): {len(output_tracks_for_this_sensor)} Tracks zwischengespeichert.") # Geändert

            print(f"    DEBUG B1 SENSOR: Sensor {sensor_name} - Alle Frames verarbeitet. Anzahl Tracks für Ziel-Frame {target_sample_token_for_output}: {len(output_tracks_for_this_sensor)}") # Hinzugefügt
            output_filename = f"tracks_{sensor_name}_target_{target_sample_token_for_output}.json"
            output_filepath = intermediate_tracks_base_dir / output_filename
            
            print(f"    DEBUG B1 SENSOR: Versuche JSON zu schreiben: {output_filepath}") # Hinzugefügt
            with open(output_filepath, "w") as f:
                json.dump(output_tracks_for_this_sensor, f, indent=2, default=_to_json_serializable)
            print(f"    INFO B1 SENSOR: Finale Tracks für Sensor {sensor_name} (für Token {target_sample_token_for_output}) gespeichert in: {output_filepath}") # Geändert

        print(f"\n✓ B1: Tracking für alle simulierten Sensoren abgeschlossen.")

    except Exception as e: # Umfassender try-except Block
        print(f"FEHLER B1: Ein unerwarteter Fehler ist in main() aufgetreten: {type(e).__name__} - {e}")
        print("------- STACKTRACE -------")
        traceback.print_exc()
        print("--------------------------")
        print("DEBUG B1: Skript wird aufgrund des Fehlers beendet.")

if __name__ == "__main__":
    main()