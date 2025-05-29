#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
from typing import List, Dict, Any, Optional

from oft.utils.config import load_config
from oft.fusion.nms_3d import nms_bev_3d 
from oft.utils.common_utils import _to_json_serializable 
# Für Token-Lookup, kann entweder das neue oder alte Dataset sein, da nur Samples-Liste gebraucht wird
from oft.data.dataset import TruckScenesDataset 

def load_sensor_detections_for_token(
    filepath: str, 
    target_token: str
) -> List[Dict[str, Any]]:
    """Lädt alle Detektionen für einen spezifischen Sample-Token aus einer JSON-Datei."""
    detections_for_token = []
    if not os.path.exists(filepath):
        print(f"    WARNUNG Stage 4 (load_sensor_detections): Datei nicht gefunden: {filepath}")
        return detections_for_token
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        for det in data: # A0 speichert eine Liste von Detektionen
            if det.get("sample_token") == target_token:
                detections_for_token.append(det)
    except json.JSONDecodeError:
        print(f"    FEHLER Stage 4 (load_sensor_detections): Konnte JSON nicht parsen: {filepath}")
    except Exception as e:
        print(f"    FEHLER Stage 4 (load_sensor_detections): Unerwarteter Fehler beim Laden von {filepath}: {e}")
    return detections_for_token

def main():
    parser = argparse.ArgumentParser(description="Stage 4: Grouped NMS Fusion for Simulated Sensor Detections")
    parser.add_argument("-c","--pipeline", required=False, default="config/pipeline.yaml",
                        help="Pfad zur pipeline.yaml.")
    args = parser.parse_args()

    cfg    = load_config(args.pipeline)
    dcfg   = cfg["dataset"]
    ocfg   = cfg["output"]
    fusion_cfg = cfg["fusion"]
    sim_cfg = cfg.get("simulation", {})
    vcfg = cfg["visualization"]

    target_sample_idx = int(vcfg.get("sample_idx", 0))
    
    temp_ds = TruckScenesDataset( # Für Token-Lookup
        dataroot=str(dcfg["dataroot"]), 
        version=str(dcfg["version"]), 
        history_window=0, max_boxes=0, augment_noise_std=0.0
    )
    if target_sample_idx >= len(temp_ds.samples):
        print(f"FEHLER Stage 4: visualization.sample_idx ({target_sample_idx}) ist außerhalb des Dataset-Bereichs.")
        return
    target_sample_token = temp_ds.samples[target_sample_idx]
    del temp_ds

    print(f"INFO Stage 4: Erzeuge EINE fusionierte Box pro GT-Objekt für Sample-Token: {target_sample_token}")

    simulated_detections_base_dir = ocfg.get("simulated_detections_dir", "/output/simulated_detections")
    
    # Annahme: Wir haben genau drei definierte Quellen, die wir pro Objekt fusionieren wollen.
    # Diese müssen in der pipeline.yaml unter simulation.sources definiert sein
    # und ihre Schlüssel in fusion.active_simulated_sources_for_fusion aufgelistet sein.
    # Für dieses spezielle Szenario nehmen wir an, es sind die drei typischen:
    # sensor_gt_clean_dropout, sensor_gt_noisy_A, sensor_gt_noisy_B_dropout
    # Wichtig: Die Reihenfolge hier ist relevant, wenn A0 Dropout anwendet.
    # Wir lesen die zu fusionierenden Quellen aus der Config, um flexibel zu bleiben.
    
    sources_to_fuse_keys = fusion_cfg.get("active_simulated_sources_for_fusion", [])
    if len(sources_to_fuse_keys) == 0 : # Geändert von !=3 zu ==0 um flexibler zu sein, aber Ideal sind 3
        print(f"FEHLER Stage 4: Bitte definieren Sie 'active_simulated_sources_for_fusion' (idealerweise 3 Quellen) in der pipeline.yaml (fusion Sektion).")
        active_sim_sources_data = [] # Leere Liste um Folgefehler zu vermeiden
    else:
        print(f"  INFO Stage 4: Verwende folgende {len(sources_to_fuse_keys)} Quellen für die gruppenweise Fusion: {sources_to_fuse_keys}")


    active_sim_sources_data: List[Optional[List[Dict[str,Any]]]] = [None] * len(sources_to_fuse_keys)
    max_detections_in_any_source = 0

    for idx, sensor_key_name in enumerate(sources_to_fuse_keys):
        sensor_config_details = sim_cfg.get("sources", {}).get(sensor_key_name)
        if not sensor_config_details:
            print(f"    WARNUNG Stage 4: Konfiguration für simulierten Sensor '{sensor_key_name}' nicht unter 'simulation.sources' gefunden. Diese Quelle wird ignoriert.")
            continue # lasse active_sim_sources_data[idx] als None

        sim_det_filename = sensor_config_details.get("output_file")
        if not sim_det_filename:
            print(f"    WARNUNG Stage 4: Kein 'output_file' für simulierten Sensor '{sensor_key_name}' definiert. Diese Quelle wird ignoriert.")
            continue

        sim_det_filepath = os.path.join(simulated_detections_base_dir, sim_det_filename)
        detections = load_sensor_detections_for_token(sim_det_filepath, target_sample_token)
        active_sim_sources_data[idx] = detections
        if detections:
            max_detections_in_any_source = max(max_detections_in_any_source, len(detections))
        print(f"    Quelle '{sensor_key_name}': {len(detections) if detections else 0} Detektionen für Token {target_sample_token} geladen.")

    if max_detections_in_any_source == 0:
        print(f"INFO Stage 4: Keine Detektionen in irgendeiner der aktiven Quellen für Sample {target_sample_token} gefunden.")
        # Fallback: Leere JSON schreiben
        output_fused_path = ocfg.get("fused_single_box_per_gt_json", "/output/fused_single_box_per_gt.json") # Neuer Output-Dateiname
        os.makedirs(os.path.dirname(output_fused_path), exist_ok=True)
        with open(output_fused_path, "w") as f: json.dump([], f, indent=2)
        print(f"✓ Stage 4: schrieb 0 fusionierte Boxen nach {output_fused_path}")
        return

    final_fused_boxes_output = []
    iou_th = float(fusion_cfg.get("iou_threshold", 0.1)) # Standard NMS IoU

    # Iteriere basierend auf der maximalen Anzahl von Detektionen, die in einer der Quellen gefunden wurden.
    # Dies setzt voraus, dass die i-te Detektion in jeder Datei (falls vorhanden) zum selben Objekt gehört.
    print(f"  INFO Stage 4: Verarbeite bis zu {max_detections_in_any_source} Objektgruppen...")
    for i in range(max_detections_in_any_source):
        group_boxes_params = []
        group_scores = []
        group_original_sources = [] # Um die Quelle der "gewinnenden" Box zu speichern

        # Sammle die i-te Box aus jeder Sensorquelle, falls vorhanden
        for source_idx, sensor_detections in enumerate(active_sim_sources_data):
            if sensor_detections and i < len(sensor_detections):
                det = sensor_detections[i]
                box_7d = det.get("translation_world", []) + det.get("size_wlh", []) + [det.get("rotation_yaw_world", 0.0)]
                if len(box_7d) == 7:
                    group_boxes_params.append(box_7d)
                    group_scores.append(float(det.get("simulated_score", 0.1))) # Score aus A0
                    group_original_sources.append(sources_to_fuse_keys[source_idx]) # Name der Quelle

        if not group_boxes_params: # Keine Boxen für diese Gruppe gefunden
            continue

        boxes_np = np.array(group_boxes_params)
        scores_np = np.array(group_scores)
        
        if boxes_np.shape[0] == 1: # Nur eine Box in der Gruppe, kein NMS nötig
            keep_indices = [0]
        elif boxes_np.shape[0] > 1:
            # Wende NMS auf diese kleine Gruppe an
            # Die Standard-NMS sollte die Box mit dem höchsten Score behalten,
            # wenn andere stark überlappen (was sie sollten, da sie vom selben GT stammen).
            keep_indices = nms_bev_3d(boxes_np, scores_np, iou_threshold=iou_th)
        else: # Sollte nicht passieren, da wir 'if not group_boxes_params: continue' haben
            keep_indices = []

        if keep_indices:
            # Normalerweise sollte NMS genau einen Index zurückgeben, wenn alle Boxen stark überlappen
            # und von derselben Entität stammen, außer es gibt exakt gleiche Scores und IoUs.
            # Wir nehmen den ersten Index aus den Ergebnissen (der mit dem höchsten Score nach NMS).
            chosen_idx_in_group = keep_indices[0]
            
            final_box_params = boxes_np[chosen_idx_in_group]
            final_score = scores_np[chosen_idx_in_group]
            winning_source = group_original_sources[chosen_idx_in_group]

            final_fused_boxes_output.append({
                "sample_token": target_sample_token,
                "translation_world": final_box_params[0:3].tolist(),
                "size_wlh": final_box_params[3:6].tolist(),
                "rotation_yaw_world": float(final_box_params[6]),
                "fusion_score": float(final_score), # Score der Box, die NMS überlebt hat
                "original_sim_source": winning_source, # Von welcher Quelle stammt die finale Box
                "fused_box_type": "stage4_single_representation" # Neuer Typ für Klarheit
            })
        # else:
            # print(f"    DEBUG Stage 4: Für Gruppe {i} wurde keine Box nach NMS behalten.")


    output_fused_path = ocfg.get("fused_single_box_per_gt_json", "/output/fused_single_box_per_gt.json")
    output_dir_for_json = os.path.dirname(output_fused_path)
    if not os.path.exists(output_dir_for_json) and output_dir_for_json:
        os.makedirs(output_dir_for_json, exist_ok=True)

    with open(output_fused_path, "w") as f:
        json.dump(final_fused_boxes_output, f, indent=2, default=_to_json_serializable)

    print(f"✓ Stage 4: schrieb {len(final_fused_boxes_output)} einzelne fusionierte Boxen nach {output_fused_path}")

if __name__ == "__main__":
    main()