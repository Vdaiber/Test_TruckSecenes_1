#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
import random # For dropout

from oft.utils.config import load_config
from oft.data.dataset import TruckScenesDataset # To load GT data
from oft.utils.common_utils import _to_json_serializable # For saving JSON

def apply_noise_to_boxes(boxes_world_7d: np.ndarray, noise_std_actual: float) -> np.ndarray:
    """
    Applies Gaussian noise to the centers (x,y,z) of the given boxes.
    Args:
        boxes_world_7d (np.ndarray): Array of shape (N,7) [x,y,z,w,l,h,yaw] in world coords.
        noise_std_actual (float): Single float value for std dev of noise applied to x,y,z.
                                 If 0, no noise is applied.
    Returns:
        np.ndarray: Noisy boxes, same shape as input.
    """
    if noise_std_actual <= 0.0 or boxes_world_7d.shape[0] == 0:
        return boxes_world_7d.copy() # Return a copy if no noise or no boxes

    noisy_boxes = boxes_world_7d.copy()
    # Erzeuge Rauschen nur für die x,y,z Koordinaten
    noise_vectors = np.random.normal(
        loc=0.0,
        scale=noise_std_actual, # Dieser Wert ist die Standardabweichung für jede der 3 Achsen
        size=(noisy_boxes.shape[0], 3) # Noise for x,y,z
    )
    noisy_boxes[:, :3] += noise_vectors
    return noisy_boxes

def apply_dropout_to_boxes(boxes_world_7d: np.ndarray, dropout_rate: float) -> np.ndarray:
    """
    Randomly removes a fraction of boxes.
    Args:
        boxes_world_7d (np.ndarray): Array of shape (N,7).
        dropout_rate (float): Fraction of boxes to remove (0.0 to 1.0).
    Returns:
        np.ndarray: Array of boxes after dropout, shape (M,7) where M <= N.
    """
    if not (0.0 <= dropout_rate <= 1.0):
        raise ValueError(f"Dropout rate muss zwischen 0.0 und 1.0 liegen, ist aber {dropout_rate}")
    if dropout_rate == 0.0 or boxes_world_7d.shape[0] == 0:
        return boxes_world_7d.copy()
    
    num_to_drop = int(round(boxes_world_7d.shape[0] * dropout_rate))
    num_to_keep = boxes_world_7d.shape[0] - num_to_drop

    if num_to_keep <= 0:
        return np.zeros((0,7), dtype=boxes_world_7d.dtype)

    indices = np.arange(boxes_world_7d.shape[0])
    np.random.shuffle(indices) # Mische Indizes für zufälligen Dropout
    keep_indices = indices[:num_to_keep]
    return boxes_world_7d[keep_indices].copy()


def main():
    parser = argparse.ArgumentParser(description="A0: Generate Simulated Sensor Detections from GT")
    parser.add_argument("-c", "--pipeline", required=False, default="config/pipeline.yaml",
                        help="Pfad zur pipeline.yaml.")
    args = parser.parse_args()

    cfg = load_config(args.pipeline)
    dcfg = cfg["dataset"]
    vcfg = cfg["visualization"]
    ocfg = cfg["output"]
    sim_cfg = cfg.get("simulation", {})

    if not sim_cfg.get("enabled", False):
        print("INFO A0: Simulation ist in der pipeline.yaml deaktiviert. Überspringe Generierung.")
        return

    target_sample_idx = int(vcfg.get("sample_idx", 0))
    # Basis-Rauschwert aus der Dataset-Konfiguration (wird als Skalierungsfaktor verwendet)
    base_noise_std_from_config = float(dcfg.get("augment_noise_std", 0.0)) 

    print(f"INFO A0: Generiere simulierte Detektionen für Sample-Index {target_sample_idx}")

    ds_clean = TruckScenesDataset(
        dataroot=str(dcfg["dataroot"]),
        version=str(dcfg["version"]).strip(),
        history_window=0, 
        max_boxes=None, # HIER GEÄNDERT: Lade alle Boxen für A0
        augment_noise_std=0.0 # Wichtig: Saubere GT als Basis laden
    )

    if target_sample_idx >= len(ds_clean.samples):
        print(f"FEHLER A0: visualization.sample_idx ({target_sample_idx}) ist außerhalb des Dataset-Bereichs.")
        return
    
    target_sample_token = ds_clean.samples[target_sample_idx]
    item_gt = ds_clean[target_sample_idx] 
    gt_boxes_world_clean = item_gt["current"].copy() # Kopie nehmen, um Original nicht zu verändern

    if gt_boxes_world_clean.shape[0] == 0:
        print(f"WARNUNG A0: Keine GT-Boxen im Ziel-Sample {target_sample_token} gefunden. Es werden leere Dateien erzeugt.")
    else:
        print(f"  Geladene saubere GT für Token {target_sample_token}: {gt_boxes_world_clean.shape[0]} Boxen.")

    simulated_detections_base_dir = ocfg.get("simulated_detections_dir", "/output/simulated_detections")
    os.makedirs(simulated_detections_base_dir, exist_ok=True)

    simulated_sources_config = sim_cfg.get("sources", {})
    if not simulated_sources_config:
        print("WARNUNG A0: Keine 'simulation.sources' in der pipeline.yaml definiert. Keine Daten generiert.")
        return

    for sensor_name, sensor_params in simulated_sources_config.items():
        print(f"  Generiere Daten für simulierten Sensor: {sensor_name}")
        
        noise_multiplier = float(sensor_params.get("noise_std_multiplier", 1.0))
        dropout_rate = float(sensor_params.get("dropout_rate", 0.0))
        simulated_score = float(sensor_params.get("score", 0.8))
        output_filename = sensor_params.get("output_file", f"{sensor_name}.json")
        output_path = os.path.join(simulated_detections_base_dir, output_filename)

        # Starte immer mit den sauberen GT-Boxen für jeden simulierten Sensor
        current_boxes_for_sensor = gt_boxes_world_clean.copy() 

        # 1. Rauschen anwenden (nur wenn Multiplikator > 0 und Basis-Std > 0)
        actual_noise_to_apply = base_noise_std_from_config * noise_multiplier
        if actual_noise_to_apply > 0 and current_boxes_for_sensor.shape[0] > 0:
            current_boxes_for_sensor = apply_noise_to_boxes(current_boxes_for_sensor, actual_noise_to_apply)
            print(f"    Rauschen angewendet (StdDev: {actual_noise_to_apply:.2f})")
        elif current_boxes_for_sensor.shape[0] > 0:
             print(f"    Kein Rauschen angewendet (noise_std_multiplier oder base_noise_std ist 0).")


        # 2. Dropout anwenden
        if dropout_rate > 0 and current_boxes_for_sensor.shape[0] > 0:
            num_before_dropout = current_boxes_for_sensor.shape[0]
            current_boxes_for_sensor = apply_dropout_to_boxes(current_boxes_for_sensor, dropout_rate)
            print(f"    Dropout angewendet (Rate: {dropout_rate*100:.0f}%): {num_before_dropout} -> {current_boxes_for_sensor.shape[0]} Boxen")
        elif current_boxes_for_sensor.shape[0] > 0:
            print(f"    Kein Dropout angewendet (dropout_rate ist 0).")


        # 3. In JSON-Format konvertieren
        output_data_list = []
        if current_boxes_for_sensor.shape[0] > 0:
            for box_7d in current_boxes_for_sensor:
                output_data_list.append({
                    "sample_token": target_sample_token,
                    "translation_world": box_7d[0:3].tolist(),
                    "size_wlh": box_7d[3:6].tolist(), 
                    "rotation_yaw_world": float(box_7d[6]),
                    "simulated_score": simulated_score # Score aus der Config für diese Quelle
                })
        
        with open(output_path, "w") as f:
            json.dump(output_data_list, f, indent=2, default=_to_json_serializable)
        print(f"    Gespeichert: {len(output_data_list)} Detektionen in {output_path}")

    print(f"✓ A0: Generierung simulierter Detektionen abgeschlossen.")

if __name__ == "__main__":
    main()