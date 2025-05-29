#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np

from oft.utils.config     import load_config
from oft.fusion.nms_3d    import nms_bev_3d 
from oft.utils.common_utils import _to_json_serializable 

def main():
    parser = argparse.ArgumentParser(description="Stage 4: NMS Fusion on Simulated Sensor Detections")
    parser.add_argument("-c","--pipeline", required=False, default="config/pipeline.yaml",
                        help="Pfad zur pipeline.yaml.")
    args = parser.parse_args()

    cfg    = load_config(args.pipeline)
    ocfg   = cfg["output"]
    fusion_cfg = cfg["fusion"]
    sim_cfg = cfg.get("simulation", {}) # Simulationskonfiguration
    vcfg = cfg["visualization"] # Für den target_sample_token

    # Ziel-Sample-Index aus der Visualisierungskonfiguration
    target_sample_idx = int(vcfg.get("sample_idx", 0))
    
    # Temporäres Dataset nur für den Token-Lookup
    from oft.data.old_dataset import TruckScenesDataset 
    # Initialisiere mit minimalen Parametern, da wir nur den Token brauchen
    temp_ds = TruckScenesDataset(
        dataroot=str(cfg["dataset"]["dataroot"]), 
        version=str(cfg["dataset"]["version"]), 
        history_window=0, 
        max_boxes=0, # Nicht relevant für Token-Lookup
        augment_noise_std=0.0
    )
    if target_sample_idx >= len(temp_ds.samples):
        print(f"FEHLER Stage 4: visualization.sample_idx ({target_sample_idx}) ist außerhalb des Dataset-Bereichs.")
        return
    target_sample_token = temp_ds.samples[target_sample_idx]
    del temp_ds # Nicht mehr benötigt

    print(f"INFO Stage 4: Fusioniere simulierte Detektionen für Sample-Token: {target_sample_token}")

    simulated_detections_base_dir = ocfg.get("simulated_detections_dir", "/output/simulated_detections")
    
    active_sim_sources_keys = fusion_cfg.get("active_simulated_sources_for_fusion", [])
    if not active_sim_sources_keys:
        print("FEHLER Stage 4: Keine 'active_simulated_sources_for_fusion' in der pipeline.yaml (fusion Sektion) definiert.")
        # Erstelle eine leere Output-Datei, um Folgefehler in Stage 5 zu vermeiden
        output_fused_sim_path = ocfg.get("fused_simulated_json", "/output/fused_simulated_detections.json")
        os.makedirs(os.path.dirname(output_fused_sim_path), exist_ok=True)
        with open(output_fused_sim_path, "w") as f: json.dump([], f)
        print(f"✓ Stage 4: schrieb 0 fused simulated entries (keine Quellen definiert) → {output_fused_sim_path}")
        return

    all_boxes_to_fuse = []
    all_scores_to_fuse = []
    all_source_names_for_debug = [] # Um nachzuvollziehen, woher die Boxen kamen

    print(f"  Lade Detektionen von {len(active_sim_sources_keys)} simulierten Quellen...")
    for sensor_key_name in active_sim_sources_keys:
        sensor_config_details = sim_cfg.get("sources", {}).get(sensor_key_name)
        if not sensor_config_details:
            print(f"    WARNUNG Stage 4: Konfiguration für simulierten Sensor '{sensor_key_name}' nicht unter 'simulation.sources' gefunden. Überspringe.")
            continue

        sim_det_filename = sensor_config_details.get("output_file")
        if not sim_det_filename:
            print(f"    WARNUNG Stage 4: Kein 'output_file' für simulierten Sensor '{sensor_key_name}' definiert. Überspringe.")
            continue
            
        sim_det_filepath = os.path.join(simulated_detections_base_dir, sim_det_filename)
        
        if not os.path.exists(sim_det_filepath):
            print(f"    WARNUNG Stage 4: Simulierte Detektionsdatei {sim_det_filepath} für Sensor '{sensor_key_name}' nicht gefunden. Bitte zuerst A0_generate_simulated_detections.py ausführen.")
            continue

        try:
            with open(sim_det_filepath, 'r') as f:
                detections_from_this_sensor = json.load(f)
        except json.JSONDecodeError:
            print(f"    FEHLER Stage 4: Konnte JSON-Datei {sim_det_filepath} nicht parsen. Überspringe.")
            continue
        
        num_loaded_for_sensor = 0
        for det in detections_from_this_sensor:
            if det.get("sample_token") == target_sample_token:
                box_7d = det.get("translation_world", []) + det.get("size_wlh", []) + [det.get("rotation_yaw_world", 0.0)]
                if len(box_7d) == 7:
                    all_boxes_to_fuse.append(box_7d)
                    all_scores_to_fuse.append(float(det.get("simulated_score", 0.5))) 
                    all_source_names_for_debug.append(sensor_key_name) # Für Debugging
                    num_loaded_for_sensor += 1
        print(f"    Von '{sensor_key_name}' ({sim_det_filepath}): {num_loaded_for_sensor} Boxen für Sample {target_sample_token} geladen.")

    if not all_boxes_to_fuse:
        print(f"INFO Stage 4: Keine Boxen von den simulierten Quellen für Sample {target_sample_token} zum Fusionieren vorhanden.")
        fused_boxes_details_output = []
    else:
        boxes_to_nms_np = np.array(all_boxes_to_fuse, dtype=np.float32)
        scores_for_nms_np = np.array(all_scores_to_fuse, dtype=np.float32)
        source_names_np = np.array(all_source_names_for_debug) # Für Debugging nach NMS

        iou_th = float(fusion_cfg.get("iou_threshold", 0.1))

        print(f"INFO Stage 4: Insgesamt {boxes_to_nms_np.shape[0]} Boxen von allen simulierten Quellen werden für NMS vorbereitet.")
        
        # Debug: Zeige einige Eingaben für NMS
        # print(f"    DEBUG NMS Input Boxes (erste 5): \n{boxes_to_nms_np[:5]}")
        # print(f"    DEBUG NMS Input Scores (erste 5): {scores_for_nms_np[:5]}")
        # print(f"    DEBUG NMS Input Source Names (erste 5): {source_names_np[:5]}")


        keep_indices = nms_bev_3d(boxes_to_nms_np, scores_for_nms_np, iou_threshold=iou_th)
        
        fused_boxes_after_nms = boxes_to_nms_np[keep_indices]
        fused_scores_after_nms = scores_for_nms_np[keep_indices] 
        fused_source_names_after_nms = source_names_np[keep_indices] # Für Debugging
        
        print(f"  Nach NMS: {fused_boxes_after_nms.shape[0]} Boxen beibehalten.")
        
        fused_boxes_details_output = []
        for i in range(fused_boxes_after_nms.shape[0]):
            b_world = fused_boxes_after_nms[i]
            fused_boxes_details_output.append({
                "sample_token": target_sample_token, 
                "translation_world": [float(b_world[0]), float(b_world[1]), float(b_world[2])],
                "size_wlh":        [float(b_world[3]), float(b_world[4]), float(b_world[5])], 
                "rotation_yaw_world":    float(b_world[6]),
                "fusion_score": float(fused_scores_after_nms[i]),
                "original_sim_source": fused_source_names_after_nms[i] # Debug-Info
            })

    output_fused_sim_path = ocfg.get("fused_simulated_json", "/output/fused_simulated_detections.json")
    output_dir_for_json = os.path.dirname(output_fused_sim_path)
    if not os.path.exists(output_dir_for_json) and output_dir_for_json:
        os.makedirs(output_dir_for_json, exist_ok=True)

    with open(output_fused_sim_path, "w") as f:
        json.dump(fused_boxes_details_output, f, indent=2, default=_to_json_serializable)

    print(f"✓ Stage 4: wrote {len(fused_boxes_details_output)} fused simulated entries → {output_fused_sim_path}")

if __name__ == "__main__":
    main()