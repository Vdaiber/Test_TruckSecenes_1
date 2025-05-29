#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
import random # For dropout
from pathlib import Path 

from oft.utils.config import load_config
from oft.data.dataset import TruckScenesDataset 
from oft.utils.common_utils import _to_json_serializable 

def apply_noise_to_boxes(boxes_world_7d: np.ndarray, noise_std_actual: float) -> np.ndarray:
    if noise_std_actual <= 0.0 or boxes_world_7d.shape[0] == 0:
        return boxes_world_7d.copy()
    noisy_boxes = boxes_world_7d.copy()
    noise_vectors = np.random.normal(
        loc=0.0, scale=noise_std_actual, size=(noisy_boxes.shape[0], 3) 
    )
    noisy_boxes[:, :3] += noise_vectors
    return noisy_boxes

def apply_dropout_and_get_indices(all_boxes_7d: np.ndarray, dropout_rate: float):
    if not (0.0 <= dropout_rate <= 1.0):
        raise ValueError(f"Dropout rate muss zwischen 0.0 und 1.0 liegen, ist aber {dropout_rate}")
    n_total_boxes = all_boxes_7d.shape[0]
    if dropout_rate == 0.0 or n_total_boxes == 0:
        return all_boxes_7d.copy(), np.arange(n_total_boxes, dtype=int)
    num_to_keep = round(n_total_boxes * (1.0 - dropout_rate))
    if num_to_keep <= 0:
        return np.zeros((0,7), dtype=all_boxes_7d.dtype), np.array([], dtype=int)
    all_indices = np.arange(n_total_boxes, dtype=int)
    np.random.shuffle(all_indices) 
    keep_indices_shuffled = all_indices[:int(num_to_keep)]
    keep_indices_sorted = np.sort(keep_indices_shuffled)
    kept_boxes = all_boxes_7d[keep_indices_sorted].copy()
    return kept_boxes, keep_indices_sorted

def main():
    parser = argparse.ArgumentParser(description="A1: Generate SEQUENTIAL Simulated Sensor Detections from GT")
    parser.add_argument("-c", "--pipeline", required=False, default="config/pipeline.yaml",
                        help="Pfad zur pipeline.yaml.")
    args = parser.parse_args()

    cfg = load_config(args.pipeline) 
    dcfg = cfg["dataset"]
    vcfg = cfg["visualization"]
    ocfg = cfg["output"]
    sim_cfg = cfg.get("simulation", {})

    if not sim_cfg.get("enabled", False):
        print("INFO A1: Simulation ist in der pipeline.yaml deaktiviert. Überspringe Generierung.")
        return

    target_sample_idx = int(vcfg.get("sample_idx", 0))
    history_window_for_sequence = int(vcfg.get("history_window", 0)) 
    base_noise_std_from_config = float(dcfg.get("augment_noise_std", 0.0)) 

    print(f"INFO A1: Initialisiere TruckScenesDataset für sequenzielle Detektionsgenerierung...")
    ds_clean = TruckScenesDataset(
        dataroot=str(dcfg["dataroot"]),
        version=str(dcfg["version"]).strip(),
        history_window=int(dcfg.get("history_window_for_velocity_calc", 1)), 
        max_boxes=int(dcfg.get("gt_max_boxes") or 0) if dcfg.get("gt_max_boxes") is not None else None,
        augment_noise_std=0.0 
    )

    if not (0 <= target_sample_idx < len(ds_clean.samples)):
        print(f"FEHLER A1: visualization.sample_idx ({target_sample_idx}) ist außerhalb des Dataset-Bereichs (0-{len(ds_clean.samples)-1}).")
        return
    
    sample_tokens_in_sequence = []
    current_token_for_seq_build = ds_clean.samples[target_sample_idx] 
    num_frames_in_sequence = history_window_for_sequence + 1
    for _ in range(num_frames_in_sequence): 
        sample_tokens_in_sequence.append(current_token_for_seq_build)
        sample_record = ds_clean.ts.get("sample", current_token_for_seq_build) 
        if not sample_record: break
        prev_token = sample_record["prev"] 
        if not prev_token: break
        current_token_for_seq_build = prev_token
    sample_tokens_in_sequence.reverse() 

    if not sample_tokens_in_sequence:
        print("FEHLER A1: Keine Sample-Tokens für die Sequenz gefunden.")
        return

    print(f"INFO A1: Generiere simulierte Detektionen für {len(sample_tokens_in_sequence)} Frames, endend mit Index {target_sample_idx} (Token: {sample_tokens_in_sequence[-1]})")
    print(f"INFO A1: Verwendete Sample-Tokens in der Sequenz: {sample_tokens_in_sequence}")

    general_output_dir = Path(ocfg.get("output_dir", "output")) 
    sequential_sim_base_dir = general_output_dir / "simulated_detections_sequence"
    sequential_sim_base_dir.mkdir(parents=True, exist_ok=True) 
    print(f"INFO A1: Ausgaben werden in '{sequential_sim_base_dir.resolve()}' gespeichert.")

    simulated_sources_config = sim_cfg.get("sources", {})
    if not simulated_sources_config:
        print("WARNUNG A1: Keine 'simulation.sources' in der pipeline.yaml definiert. Keine Daten generiert.")
        return

    for current_frame_token in sample_tokens_in_sequence:
        print(f"\n  Verarbeite Frame (Sample-Token): {current_frame_token}")
        
        try:
            idx_for_current_frame = ds_clean.samples.index(current_frame_token) 
            item_gt = ds_clean[idx_for_current_frame] 
        except ValueError:
            print(f"FEHLER A1: Sample-Token {current_frame_token} nicht in ds_clean.samples gefunden. Überspringe Frame.")
            continue
            
        # Master-Kopien der Daten für diesen Frame
        master_gt_boxes_7d = item_gt["current"].copy() 
        master_gt_velocities_3d = item_gt.get("velocities", np.zeros((master_gt_boxes_7d.shape[0],3), dtype=np.float32)).copy()
        
        master_category_names = []
        num_master_gt_boxes = master_gt_boxes_7d.shape[0]
        if num_master_gt_boxes > 0:
            annotation_tokens = item_gt.get("anns", [])
            if len(annotation_tokens) == num_master_gt_boxes:
                for ann_token in annotation_tokens:
                    try:
                        # Vereinfachte Kategorie-Extraktion, da item_gt bereits alles enthält
                        annotation_record = item_gt["annotation"].get(ann_token)
                        instance_token_from_ann = annotation_record.get("instance_token") if annotation_record else None
                        instance_record = item_gt["instance"].get(instance_token_from_ann) if instance_token_from_ann else None
                        category_token = instance_record.get("category_token") if instance_record else None
                        category_record = ds_clean.ts.get("category", category_token) if category_token else None
                        master_category_names.append(category_record["name"] if category_record and "name" in category_record else "unknown")
                    except Exception:
                        master_category_names.append("unknown") 
            else:
                master_category_names = ["unknown"] * num_master_gt_boxes
        
        print(f"    DEBUG A1: Master GT für Frame {current_frame_token}: {num_master_gt_boxes} Boxen.")

        if num_master_gt_boxes == 0: 
            print(f"    INFO A1: Keine GT-Boxen im Frame {current_frame_token} (master_gt_boxes_7d ist leer). Leere JSONs werden erzeugt.")
        else:
            print(f"    Geladene saubere GT für Token {current_frame_token}: {num_master_gt_boxes} Boxen.")

        for sensor_name, sensor_params in simulated_sources_config.items():
            print(f"    Generiere Daten für simulierten Sensor: {sensor_name}")
            
            sensor_specific_output_dir = sequential_sim_base_dir / sensor_name
            sensor_specific_output_dir.mkdir(parents=True, exist_ok=True)
            output_filename_for_frame = f"{current_frame_token}.json" 
            output_path = sensor_specific_output_dir / output_filename_for_frame

            # Arbeitskopien für diesen Sensor
            boxes_to_process = master_gt_boxes_7d.copy()
            velocities_to_process = master_gt_velocities_3d.copy()
            categories_to_process = list(master_category_names)
            
            num_boxes_before_noise_dropout = boxes_to_process.shape[0]

            noise_multiplier = float(sensor_params.get("noise_std_multiplier", 1.0))
            actual_noise_to_apply = base_noise_std_from_config * noise_multiplier
            if actual_noise_to_apply > 0 and boxes_to_process.shape[0] > 0:
                boxes_to_process = apply_noise_to_boxes(boxes_to_process, actual_noise_to_apply)
            
            dropout_rate = float(sensor_params.get("dropout_rate", 0.0))
            
            if dropout_rate > 0 and boxes_to_process.shape[0] > 0:
                # Wichtig: apply_dropout_and_get_indices erhält die Boxen *vor* dem Dropout für diesen Sensor
                # und gibt die gedroptouten Boxen sowie die Indizes bezogen auf die Eingabe zurück.
                boxes_after_dropout, kept_indices = apply_dropout_and_get_indices(boxes_to_process, dropout_rate)
                
                # Filter Velocities und Kategorien basierend auf kept_indices,
                # angewendet auf die Versionen *vor* dem Dropout dieses Sensors.
                if kept_indices.size > 0 :
                    if velocities_to_process.shape[0] == num_boxes_before_noise_dropout:
                         velocities_after_dropout = velocities_to_process[kept_indices]
                    else: # Sollte nicht passieren, wenn Logik korrekt ist
                        print(f"      WARNUNG A1 (Sensor {sensor_name}): Velocities Shape Mismatch vor Dropout-Filterung. Erw: {num_boxes_before_noise_dropout}, Ist: {velocities_to_process.shape[0]}")
                        velocities_after_dropout = np.zeros((boxes_after_dropout.shape[0], 3), dtype=np.float32)
                    
                    if len(categories_to_process) == num_boxes_before_noise_dropout:
                        categories_after_dropout = [categories_to_process[i] for i in kept_indices]
                    else: # Sollte nicht passieren
                        print(f"      WARNUNG A1 (Sensor {sensor_name}): Categories Längen-Mismatch vor Dropout-Filterung. Erw: {num_boxes_before_noise_dropout}, Ist: {len(categories_to_process)}")
                        categories_after_dropout = ["unknown"] * boxes_after_dropout.shape[0]
                else: # Alle Boxen wurden gedroppt
                    velocities_after_dropout = np.zeros((0,3), dtype=np.float32)
                    categories_after_dropout = []
                
                # Aktualisiere die Arbeitsvariablen
                boxes_to_process = boxes_after_dropout
                velocities_to_process = velocities_after_dropout
                categories_to_process = categories_after_dropout

            elif boxes_to_process.shape[0] == 0: # Keine Boxen schon vor Dropout (oder alle durch vorherige Schritte weg)
                 velocities_to_process = np.zeros((0,3), dtype=np.float32)
                 categories_to_process = []
            # else: Kein Dropout, velocities_to_process und categories_to_process bleiben wie sie sind

            # --- DEBUGGING VOR KONSISTENZCHECK ---
            print(f"      DEBUG A1 (Sensor {sensor_name}): Vor Konsistenzcheck:")
            print(f"        Shape boxes_to_process: {boxes_to_process.shape}")
            print(f"        Shape velocities_to_process: {velocities_to_process.shape}")
            print(f"        Länge categories_to_process: {len(categories_to_process)}")
            # --- ENDE DEBUGGING ---

            simulated_score = float(sensor_params.get("score", 0.8))
            detections_for_json_list = []

            if boxes_to_process.shape[0] > 0: # Nur wenn Boxen vorhanden sind
                if not (boxes_to_process.shape[0] == velocities_to_process.shape[0] == len(categories_to_process)):
                    print(f"      WARNUNG A1 (Sensor {sensor_name}): Inkonsistente Anzahl nach Dropout/Verarbeitung:")
                    print(f"        Boxen: {boxes_to_process.shape[0]}, Velocities: {velocities_to_process.shape[0]}, Kategorien: {len(categories_to_process)}")
                    print(f"        Frame {current_frame_token}. Überspringe Detektionserstellung für diesen Sensor.")
                else:
                    for i in range(boxes_to_process.shape[0]):
                        box_7d = boxes_to_process[i]
                        vel_3d = velocities_to_process[i]
                        cat_name = categories_to_process[i]
                        detections_for_json_list.append({
                            "sample_token": current_frame_token, 
                            "translation_world": box_7d[0:3].tolist(),
                            "size_wlh": box_7d[3:6].tolist(), 
                            "rotation_yaw_world": float(box_7d[6]),
                            "velocity_world": vel_3d.tolist(), 
                            "detection_name": cat_name,
                            "detection_score": simulated_score 
                        })
            
            frame_output_json_content = {
                "meta": {
                    "sensor_modality": sensor_params.get("modality", "simulated"), 
                    "sensor_name": sensor_name,
                    "source_sample_token": current_frame_token, 
                    "cfg_noise_multiplier": noise_multiplier,
                    "cfg_dropout_rate": dropout_rate,
                    "cfg_simulated_score": simulated_score,
                    "num_detections_in_file": len(detections_for_json_list) # Info zur Anzahl
                },
                "results": {
                    current_frame_token: detections_for_json_list 
                }
            }
            with open(output_path, "w") as f:
                json.dump(frame_output_json_content, f, indent=2, default=_to_json_serializable)
            print(f"      JSON für Sensor {sensor_name} geschrieben, {len(detections_for_json_list)} Detektionen.")

    print(f"\n✓ A1: Generierung sequenzieller simulierter Detektionen abgeschlossen.")

if __name__ == "__main__":
    main()
