#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
from oft.utils.config import load_config

def print_simulated_detections_for_token(config_path: str, target_sample_token: str):
    cfg = load_config(config_path)
    ocfg = cfg["output"]
    sim_cfg = cfg.get("simulation", {})

    # Pfad zu den Ausgaben von A1 (sequenzielle simulierte Detektionen)
    # Stelle sicher, dass dieser Schlüssel in deiner pipeline.yaml unter output: existiert
    # oder passe den Default-Wert hier an deinen tatsächlichen Pfad an.
    sim_base_dir_str = ocfg.get("simulated_detections_dir_A1", "output/simulated_detections_sequence")
    sim_base_dir = Path(sim_base_dir_str)

    print(f"\n--- Simulierte Detektionen für Sample-Token: {target_sample_token} ---")

    simulated_sources_config = sim_cfg.get("sources", {})
    if not simulated_sources_config:
        print("  Keine 'simulation.sources' in der pipeline.yaml definiert.")
        return

    for sensor_name in simulated_sources_config.keys():
        print(f"\n  Sensor: {sensor_name}")
        filepath = sim_base_dir / sensor_name / f"{target_sample_token}.json"
        
        if not filepath.is_file():
            print(f"    Datei nicht gefunden: {filepath}")
            continue

        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Die Struktur der A1 JSON-Dateien ist:
        # {"meta": ..., "results": {sample_token: [list_of_detections]}}
        detections_list = data.get("results", {}).get(target_sample_token, [])
        
        if not detections_list:
            print("    Keine simulierten Detektionen in dieser Datei gefunden.")
            continue

        for i, det in enumerate(detections_list):
            print(f"    Det #{i+1}:")
            print(f"      Position (Welt): {det.get('translation_world')}")
            print(f"      Größe (w,l,h):   {det.get('size_wlh')}")
            yaw_value = det.get('rotation_yaw_world')
            # Stelle sicher, dass yaw_value ein numerischer Typ ist, bevor formatiert wird
            if isinstance(yaw_value, (int, float)):
                print(f"      Yaw (Welt):      {yaw_value:.3f}")
            else:
                print(f"      Yaw (Welt):      N/A (Wert: {yaw_value})")
            print(f"      Velocity (Welt): {det.get('velocity_world')}")
            print(f"      Score:           {det.get('detection_score')}")
            print(f"      Name:            {det.get('detection_name')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zeigt simulierte Detektionen von A1 für einen Sample-Token.")
    parser.add_argument("-c", "--pipeline", required=False, default="config/pipeline.yaml",
                        help="Pfad zur pipeline.yaml.")
    parser.add_argument("-t", "--token", required=True, type=str,
                        help="Der Sample-Token, dessen simulierte Detektionen angezeigt werden sollen.")
    
    script_args = parser.parse_args()
    print_simulated_detections_for_token(config_path=script_args.pipeline, target_sample_token=script_args.token)