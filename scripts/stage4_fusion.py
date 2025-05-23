#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np

from oft.utils.config     import load_config
from oft.fusion.nms_3d    import nms_bev_3d 
from oft.utils.common_utils import _to_json_serializable 

def main():
    parser = argparse.ArgumentParser(description="Stage 4: NMS Fusion on Tracks from Stage 3")
    parser.add_argument("-c","--pipeline", required=False, default="config/pipeline.yaml",
                        help="Pfad zur pipeline.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.pipeline)
    ocfg   = cfg["output"]
    fusion_cfg = cfg["fusion"]
    # vcfg = cfg["visualization"] # Nicht mehr direkt für Token-Lookup hier benötigt

    tracks_json_path = ocfg.get("tracks_json", "/output/tracks.json")
    output_fused_path = ocfg.get("fused_json", ocfg.get("dets_json", "/output/fused_detections.json"))
    iou_th    = float(fusion_cfg.get("iou_threshold", 0.3))

    print(f"INFO Stage 4: Lade Tracks aus {tracks_json_path}")
    if not os.path.exists(tracks_json_path):
        print(f"FEHLER: Tracks-Datei {tracks_json_path} nicht gefunden. Bitte zuerst Stage 3 ausführen.")
        return

    with open(tracks_json_path, 'r') as f:
        # tracks.json enthält jetzt EIN Dictionary für den Ziel-Frame
        frame_data_for_fusion = json.load(f)

    if not frame_data_for_fusion or not isinstance(frame_data_for_fusion, dict):
        print(f"FEHLER: Tracks-Datei {tracks_json_path} ist leer oder hat ein unerwartetes Format (erwartet Dictionary).")
        with open(output_fused_path, "w") as f: json.dump([], f, indent=2)
        print(f"✓ Stage 4: schrieb 0 Einträge → {output_fused_path} (Input-Tracks-Datei leer/falsch).")
        return

    loaded_sample_token = frame_data_for_fusion.get("sample_token")
    if not loaded_sample_token:
        print(f"FEHLER: Kein 'sample_token' in den geladenen Daten aus {tracks_json_path} gefunden.")
        with open(output_fused_path, "w") as f: json.dump([], f, indent=2)
        print(f"✓ Stage 4: schrieb 0 Einträge → {output_fused_path} (Kein Sample-Token im Input).")
        return
        
    print(f"INFO Stage 4: Verarbeite Tracks aus Sample-Token: {loaded_sample_token}")

    tracks_for_nms = frame_data_for_fusion.get("tracks", [])

    if not tracks_for_nms:
        print(f"INFO Stage 4: Keine Tracks im geladenen Frame ({loaded_sample_token}) für NMS vorhanden.")
        fused_boxes_details = []
    else:
        print(f"INFO Stage 4: {len(tracks_for_nms)} Tracks werden für NMS vorbereitet.")
        boxes_to_nms = np.array([track['box_world'] for track in tracks_for_nms]) 
        # Verwende 'confidence_score' falls vorhanden, sonst 'hits' als Fallback
        scores_for_nms = np.array([track.get('confidence_score', track.get('hits', 1.0)) for track in tracks_for_nms], dtype=np.float32)

        print(f"  Boxen für NMS (Shape): {boxes_to_nms.shape}")
        print(f"  Scores für NMS (Shape): {scores_for_nms.shape}")
        # print(f"  Beispiel Box (erste): {boxes_to_nms[0].tolist() if len(boxes_to_nms) > 0 else 'N/A'}")
        # print(f"  Beispiel Score (erster): {scores_for_nms[0] if len(scores_for_nms) > 0 else 'N/A'}")
        # print(f"  IoU Threshold für NMS: {iou_th}")

        keep_indices = nms_bev_3d(boxes_to_nms, scores_for_nms, iou_threshold=iou_th)
        
        fused_tracks_details = [tracks_for_nms[i] for i in keep_indices]
        print(f"  Nach NMS: {len(fused_tracks_details)} Tracks/Boxen beibehalten.")
        
        fused_boxes_details = []
        for kept_track_info in fused_tracks_details:
            b_world = kept_track_info['box_world'] 
            fused_boxes_details.append({
                "sample_token": loaded_sample_token, 
                "track_id": kept_track_info.get("track_id", -1), 
                "translation_world": [float(b_world[0]), float(b_world[1]), float(b_world[2])],
                "size_wlh":        [float(b_world[3]), float(b_world[4]), float(b_world[5])], 
                "rotation_yaw_world":    float(b_world[6]),
                "score_final_for_nms": float(kept_track_info.get('confidence_score', kept_track_info.get('hits', 0.0))),
                "hits_original": float(kept_track_info.get('hits', 0)) # Behalte auch die originalen Hits
            })

    output_dir_for_json = os.path.dirname(output_fused_path)
    if not os.path.exists(output_dir_for_json) and output_dir_for_json:
        os.makedirs(output_dir_for_json, exist_ok=True)

    with open(output_fused_path, "w") as f:
        json.dump(fused_boxes_details, f, indent=2, default=_to_json_serializable)

    print(f"✓ Stage 4: wrote {len(fused_boxes_details)} fused entries → {output_fused_path}")

if __name__ == "__main__":
    main()