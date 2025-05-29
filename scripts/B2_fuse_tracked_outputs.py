#!/usr/bin/env python3
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional 
import traceback # Hinzugefügt für besseres Fehler-Logging

# Interne Projekt-Imports
from oft.utils.config import load_config
# from oft.data.dataset import TruckScenesDataset # Wird nur für target_sample_token benötigt, ds_helper
from truckscenes import TruckScenes # Direkter Import für ts-Instanz
from oft.fusion.nms_3d import nms_bev_3d 
from oft.utils.common_utils import _to_json_serializable
from oft.utils.fusion_utils import calculate_track_distance, fuse_track_group

def main():
    print("DEBUG B2: main() gestartet.")
    parser = argparse.ArgumentParser(description="B2: Fuse Tracked Outputs (Advanced using fusion_utils)")
    parser.add_argument("-c", "--pipeline", required=False, default="config/pipeline.yaml",
                        help="Pfad zur pipeline.yaml.")
    args = parser.parse_args()

    try:
        cfg = load_config(args.pipeline)
        dcfg = cfg["dataset"]
        vcfg = cfg["visualization"]
        ocfg = cfg["output"]
        sim_cfg = cfg.get("simulation", {})
        fusion_cfg = cfg.get("fusion", {})
        
        track_fusion_max_dist = float(fusion_cfg.get("track_to_track_max_dist", 2.0))
        use_velocity_in_track_dist = bool(fusion_cfg.get("use_velocity_in_track_dist", False))
        velocity_weight_in_track_dist = float(fusion_cfg.get("velocity_weight_in_track_dist", 0.5))
        min_confidence_for_b2_input = float(fusion_cfg.get("min_confidence_for_b2_input", 0.0)) 

        print(f"DEBUG B2: Konfiguration geladen. min_confidence_for_b2_input={min_confidence_for_b2_input}, track_to_track_max_dist={track_fusion_max_dist}, use_velocity={use_velocity_in_track_dist}, vel_weight={velocity_weight_in_track_dist}")

        if not sim_cfg.get("enabled", False) or not fusion_cfg.get("enabled", False):
            print("INFO B2: Simulation oder Fusion ist in der pipeline.yaml deaktiviert. Überspringe B2.")
            return

        intermediate_tracks_base_dir_str = ocfg.get("intermediate_tracks_dir_B1", "output/intermediate_tracks")
        intermediate_tracks_base_dir = Path(intermediate_tracks_base_dir_str)

        final_fused_tracks_dir_str = ocfg.get("final_fused_tracks_dir_B2", "output/final_fused_tracks")
        final_fused_tracks_dir = Path(final_fused_tracks_dir_str)
        final_fused_tracks_dir.mkdir(parents=True, exist_ok=True)

        # NEU: Output-Verzeichnis für die B1-Tracks (Input für B2-Gruppierung)
        b1_inputs_for_b2_dir_str = ocfg.get("b1_inputs_for_b2_dir", "output/intermediate_b1_tracks_for_b2_input")
        b1_inputs_for_b2_dir = Path(b1_inputs_for_b2_dir_str)
        b1_inputs_for_b2_dir.mkdir(parents=True, exist_ok=True)
        print(f"DEBUG B2: B1-Input-Tracks für B2 werden in '{b1_inputs_for_b2_dir.resolve()}' gespeichert.")


        target_sample_idx = int(vcfg.get("sample_idx", 0))
        nms_iou_threshold = float(fusion_cfg.get("iou_threshold", 0.1))

        print(f"INFO B2: Initialisiere TruckScenes für target_sample_token...")
        # ds_helper wurde nur für target_sample_token benötigt, wir können ts direkt verwenden
        ts = TruckScenes(version=str(dcfg["version"]).strip(), dataroot=str(dcfg["dataroot"]))
        
        if not (0 <= target_sample_idx < len(ts.sample)):
            print(f"FEHLER B2: visualization.sample_idx ({target_sample_idx}) ist außerhalb des gültigen Bereichs.")
            return
        target_sample_token = ts.sample[target_sample_idx]["token"]
        print(f"INFO B2: Target Sample Token für Fusion: {target_sample_token}")

        all_tracks_from_all_sensors_raw: List[Dict[str, Any]] = [] 
        simulated_sources_config: Dict[str, Any] = sim_cfg.get("sources", {})

        for sensor_name in simulated_sources_config.keys():
            track_file_name = f"tracks_{sensor_name}_target_{target_sample_token}.json"
            track_file_path = intermediate_tracks_base_dir / track_file_name
            if track_file_path.is_file():
                print(f"  Lade Tracks für Sensor {sensor_name} aus {track_file_path}")
                with open(track_file_path, 'r') as f:
                    tracks_this_sensor = json.load(f)
                    for i, track in enumerate(tracks_this_sensor):
                        track["source_sensor"] = sensor_name
                        # original_track_id_in_source ist die ID, die der B1-Tracker diesem Track für diesen Sensor gegeben hat
                        track["original_track_id_in_source"] = track.get("track_id") 
                        track["unique_id_before_fusion"] = f"{sensor_name}_{track.get('track_id', i)}" 
                    all_tracks_from_all_sensors_raw.extend(tracks_this_sensor)
                    print(f"    {len(tracks_this_sensor)} Tracks geladen.")
            else:
                print(f"  WARNUNG B2: Track-Datei für Sensor {sensor_name} nicht gefunden: {track_file_path}")

        if not all_tracks_from_all_sensors_raw:
            print("WARNUNG B2: Keine Tracks von B1 zum Fusionieren gefunden (vor Filterung).")
            # Schreibe trotzdem eine leere Datei für nachfolgende Schritte, falls erwartet
            output_filename_empty = f"fused_tracks_advanced_target_{target_sample_token}.json"
            output_filepath_empty = final_fused_tracks_dir / output_filename_empty
            with open(output_filepath_empty, "w") as f: json.dump([], f, indent=2, default=_to_json_serializable)
            
            # NEU: Schreibe auch eine leere Datei für die B1-Inputs
            b1_input_filename_empty = f"b1_tracks_for_b2_input_target_{target_sample_token}.json"
            b1_input_filepath_empty = b1_inputs_for_b2_dir / b1_input_filename_empty
            with open(b1_input_filepath_empty, "w") as f: json.dump([], f, indent=2, default=_to_json_serializable)
            return
            
        print(f"INFO B2: Insgesamt {len(all_tracks_from_all_sensors_raw)} Tracks von allen Sensoren gesammelt (vor Filterung).")

        print(f"DEBUG B2: Filtere Tracks für B2-Input mit min_confidence_score: {min_confidence_for_b2_input}")
        all_tracks_from_all_sensors = [
            t for t in all_tracks_from_all_sensors_raw if t.get("confidence_score", 0.0) >= min_confidence_for_b2_input
        ]
        print(f"INFO B2: Nach Konfidenz-Filterung: {len(all_tracks_from_all_sensors)} von {len(all_tracks_from_all_sensors_raw)} Tracks übrig für Fusion.")

        # NEU: Speichere die gefilterten B1-Tracks, die als Input für die Gruppierung dienen
        if all_tracks_from_all_sensors: # Nur speichern, wenn etwas übrig ist
            b1_input_filename = f"b1_tracks_for_b2_input_target_{target_sample_token}.json"
            b1_input_filepath = b1_inputs_for_b2_dir / b1_input_filename
            print(f"DEBUG B2: Speichere {len(all_tracks_from_all_sensors)} gefilterte B1-Tracks (Input für Gruppierung) nach: {b1_input_filepath}")
            with open(b1_input_filepath, "w") as f:
                json.dump(all_tracks_from_all_sensors, f, indent=2, default=_to_json_serializable)
        elif not all_tracks_from_all_sensors_raw: # Fall, wo schon vorher nichts da war, oben behandelt
            pass
        else: # Fall, wo nach Filterung nichts übrig ist, aber vorher Tracks da waren
            b1_input_filename_empty = f"b1_tracks_for_b2_input_target_{target_sample_token}.json"
            b1_input_filepath_empty = b1_inputs_for_b2_dir / b1_input_filename_empty
            with open(b1_input_filepath_empty, "w") as f: json.dump([], f, indent=2, default=_to_json_serializable)
            print(f"DEBUG B2: Nach Konfidenzfilter keine B1-Tracks übrig. Leere Datei gespeichert: {b1_input_filepath_empty}")


        if not all_tracks_from_all_sensors:
            print("WARNUNG B2: Keine Tracks nach Konfidenz-Filterung für Fusion übrig.")
            output_filename_empty = f"fused_tracks_advanced_target_{target_sample_token}.json"
            output_filepath_empty = final_fused_tracks_dir / output_filename_empty
            with open(output_filepath_empty, "w") as f: json.dump([], f, indent=2, default=_to_json_serializable)
            return

        final_merged_tracks: List[Dict[str, Any]] = []
        all_tracks_from_all_sensors.sort(key=lambda t: t.get("confidence_score", 0.0), reverse=True)
        processed_mask = [False] * len(all_tracks_from_all_sensors)

        print(f"\nDEBUG B2: Starte iterative Fusion und Gruppierung für {len(all_tracks_from_all_sensors)} Tracks...")
        for i in range(len(all_tracks_from_all_sensors)):
            if processed_mask[i]:
                continue
            ref_track = all_tracks_from_all_sensors[i]
            # print(f"  DEBUG B2: Iteration {i+1}/{len(all_tracks_from_all_sensors)}. Referenz-Track: {ref_track.get('unique_id_before_fusion')} (Score: {ref_track.get('confidence_score', 0.0):.2f})")
            
            current_fusion_group = [ref_track] 
            added_candidates_debug: List[str] = []
            
            for j in range(len(all_tracks_from_all_sensors)): 
                if i == j or processed_mask[j]: 
                    continue
                candidate_track = all_tracks_from_all_sensors[j]
                if ref_track.get("source_sensor") == candidate_track.get("source_sensor"):
                    continue 
                distance = calculate_track_distance(ref_track, candidate_track,
                                                    use_velocity_term=use_velocity_in_track_dist,
                                                    velocity_weight=velocity_weight_in_track_dist)
                if distance < track_fusion_max_dist:
                    current_fusion_group.append(candidate_track)
                    added_candidates_debug.append(f"{candidate_track.get('unique_id_before_fusion')} (Dist: {distance:.2f}m)")
            
            # if len(current_fusion_group) > 1:
            #     print(f"    DEBUG B2:  -> Fusion-Gruppe für Ref {ref_track.get('unique_id_before_fusion')} gebildet mit: {added_candidates_debug}")
            # else:
            #     print(f"    DEBUG B2:  -> Keine Fusionspartner für Ref {ref_track.get('unique_id_before_fusion')} gefunden (innerhalb {track_fusion_max_dist}m).")

            if current_fusion_group: 
                for track_in_group in current_fusion_group:
                    original_list_idx = -1
                    uid_to_find = track_in_group.get("unique_id_before_fusion")
                    if uid_to_find is not None:
                        for k_idx, t_orig in enumerate(all_tracks_from_all_sensors):
                            if t_orig.get("unique_id_before_fusion") == uid_to_find:
                                original_list_idx = k_idx
                                break
                    if original_list_idx != -1:
                        processed_mask[original_list_idx] = True
                    # else:
                        # print(f"WARNUNG B2: Konnte Track aus Gruppe (ID: {track_in_group.get('unique_id_before_fusion')}) nicht in Track-Liste zum Markieren finden.")

                merged_track = fuse_track_group(current_fusion_group) 
                if merged_track:
                    # print(f"    DEBUG B2:  --> Gruppe fusioniert zu Track ID {merged_track.get('track_id')}, Score: {merged_track.get('confidence_score',0.0):.2f}, NumContrib: {merged_track.get('num_fused_tracks')}")
                    final_merged_tracks.append(merged_track)
        
        print(f"\nINFO B2: Nach iterativer Fusion und Gruppierung: {len(final_merged_tracks)} Tracks.")

        if not final_merged_tracks:
            print("WARNUNG B2: Keine Tracks nach Fusionsschritt für NMS übrig.")
            output_filename_empty = f"fused_tracks_advanced_target_{target_sample_token}.json"
            output_filepath_empty = final_fused_tracks_dir / output_filename_empty
            with open(output_filepath_empty, "w") as f: json.dump([], f, indent=2, default=_to_json_serializable)
            return

        boxes_for_nms_list = []
        scores_for_nms_list = []
        valid_tracks_for_nms = [] 

        for track in final_merged_tracks:
            box_world_data = track.get("box_world")
            if isinstance(box_world_data, list) and len(box_world_data) == 7:
                boxes_for_nms_list.append(box_world_data)
                scores_for_nms_list.append(track.get("confidence_score", 0.0))
                valid_tracks_for_nms.append(track) 
            # else:
                # print(f"WARNUNG B2: Ungültige box_world im fusionierten Track: ID {track.get('track_id', 'N/A')}")

        if not valid_tracks_for_nms: 
            print("WARNUNG B2: Keine validen Tracks für NMS übrig nach Filterung.")
            output_filename_empty = f"fused_tracks_advanced_target_{target_sample_token}.json"
            output_filepath_empty = final_fused_tracks_dir / output_filename_empty
            with open(output_filepath_empty, "w") as f: json.dump([], f, indent=2, default=_to_json_serializable)
            return

        boxes_for_nms_np = np.array(boxes_for_nms_list, dtype=np.float32)
        scores_for_nms_np = np.array(scores_for_nms_list, dtype=np.float32)
        final_tracks_after_nms = valid_tracks_for_nms 

        if boxes_for_nms_np.ndim == 2 and boxes_for_nms_np.shape[0] > 0 and boxes_for_nms_np.shape[1] == 7:
            print(f"INFO B2: Führe finale NMS für {boxes_for_nms_np.shape[0]} fusionierte Tracks mit IoU-Schwelle {nms_iou_threshold} durch...")
            kept_indices_after_nms = nms_bev_3d(
                boxes=boxes_for_nms_np,
                scores=scores_for_nms_np,
                iou_threshold=nms_iou_threshold
            )
            print(f"INFO B2: Finale NMS abgeschlossen. {len(kept_indices_after_nms)} Tracks wurden beibehalten.")
            temp_final_tracks_after_nms: List[Dict[str, Any]] = []
            for idx in kept_indices_after_nms:
                temp_final_tracks_after_nms.append(valid_tracks_for_nms[idx])
            final_tracks_after_nms = temp_final_tracks_after_nms
        # else:
            # print(f"INFO B2: Ungültige Form oder keine Boxen für NMS ({boxes_for_nms_np.shape}). NMS wird übersprungen.")
        
        output_filename = f"fused_tracks_advanced_target_{target_sample_token}.json"
        output_filepath = final_fused_tracks_dir / output_filename
        with open(output_filepath, "w") as f:
            json.dump(final_tracks_after_nms, f, indent=2, default=_to_json_serializable) 
        print(f"INFO B2: Finale (komplex) fusionierte Tracks ({len(final_tracks_after_nms)}) gespeichert in: {output_filepath}")
        print(f"✓ B2: Komplexe Fusion der Tracks abgeschlossen.")

    except Exception as e: 
        print(f"FEHLER B2: Ein unerwarteter Fehler ist in main() aufgetreten: {type(e).__name__} - {e}")
        print("------- STACKTRACE -------")
        traceback.print_exc()
        print("--------------------------")
        print("DEBUG B2: Skript wird aufgrund des Fehlers beendet.")

if __name__ == "__main__":
    main()