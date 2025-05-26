#!/usr/bin/env python3
"""
Stage 1: Visualize Ground Truth Annotations.
Integriert die Logik aus examples/visualize_ground_truth.py und examples/visualize.py (GT-Teil).
"""
import argparse
import os
from truckscenes import TruckScenes
from truckscenes.utils.geometry_utils import BoxVisibility

from oft.utils.config import load_config
# Annahme: render_sample_boxes kommt aus oft.utils.visualization, wie im Original-Example-Skript
# Wenn es stattdessen oft.utils.sensor_utils.render_sample_boxes sein soll und die Signatur passt,
# dann diesen Import anpassen. Basierend auf dem Example-Code:
from oft.utils.visualization import render_sample_boxes
# Falls render_sample_boxes nicht existiert oder eine andere Signatur hat, müssten wir
# die Logik von render_sample_boxes (also Bild laden, Boxen holen, draw_boxes_on_image aufrufen)
# hier direkt implementieren, wie in der vorherigen Version von stage1_visualize_gt_py_v2.

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Visualize Ground Truth")
    parser.add_argument("-c", "--pipeline", required=False, default="config/pipeline.yaml",
                        help="Pfad zur pipeline.yaml.")
    args = parser.parse_args()

    cfg = load_config(args.pipeline) 
    dcfg = cfg["dataset"]
    vcfg = cfg["visualization"]
    # ocfg = cfg["output"] # Wird von run_ground_truth intern aus cfg geholt
    # rcfg = cfg["render"] # Wird von run_ground_truth intern aus cfg geholt

    ts = TruckScenes(version=dcfg["version"].strip(), dataroot=dcfg["dataroot"])
    
    # Szenen-Selektion aus dem originalen examples/visualize.py
    raw_scenes_config = vcfg.get("scenes", [])
    if not raw_scenes_config: # leer = alle Tokens abfragen
        # Die Funktion list_scenes aus oft.utils.common_utils wird hier benötigt
        # Fügen wir sie hier der Einfachheit halber hinzu oder importieren sie.
        # Für jetzt: Annahme, es wird nur der target_sample_idx verarbeitet, wenn gesetzt.
        # Die Logik für mehrere Szenen aus dem Example wird unten angepasst.
        processed_scene_tokens = [s["token"] for s in ts.scene] # Nimm alle Szenen, wenn nichts spezifiziert
    else:
        processed_scene_tokens = []
        for entry in raw_scenes_config:
            if isinstance(entry, int): # Wenn Index gegeben ist
                if 0 <= entry < len(ts.scene):
                    processed_scene_tokens.append(ts.scene[entry]["token"])
                else:
                    print(f"WARNUNG Stage 1: Szenen-Index {entry} ist ungültig. Wird ignoriert.")
            else: # Annahme: Ist bereits ein Token-String
                processed_scene_tokens.append(str(entry))

    num_frames_config = vcfg.get("num_frames", 0) # Wie viele Frames pro Szene
    # history_window wird von run_ground_truth nicht direkt verwendet, aber an es übergeben
    history_window_config = vcfg.get("history_window", 0) 


    target_sample_idx = vcfg.get("sample_idx", -1) 

    if target_sample_idx >= 0: 
        if target_sample_idx >= len(ts.sample):
            print(f"FEHLER Stage 1: visualization.sample_idx ({target_sample_idx}) ist außerhalb des Dataset-Bereichs ({len(ts.sample)} Samples).")
            return
        
        sample_to_process_token = ts.sample[target_sample_idx]["token"]
        scene_of_sample_token = ts.get("sample", sample_to_process_token)["scene_token"]
        
        print(f"INFO Stage 1: Visualisiere GT für spezifischen Sample-Index {target_sample_idx} (Token: {sample_to_process_token}) in Szene {scene_of_sample_token}")
        
        # Rufe die Logik von run_ground_truth für diese eine Szene und diesen einen Frame auf
        # run_ground_truth erwartet eine Liste von Szenen-Tokens.
        # Und num_frames=1, um nur diesen einen Frame zu verarbeiten.
        run_ground_truth_logic(ts, cfg, [scene_of_sample_token], 
                               target_sample_token_override=sample_to_process_token,
                               num_frames_override=1) # Verarbeite nur 1 Frame (den target_sample_token)
    else: 
        print(f"INFO Stage 1: Visualisiere GT für {num_frames_config if num_frames_config > 0 else 'alle'} Frames pro ausgewählter Szene ({len(processed_scene_tokens)} Szenen).")
        run_ground_truth_logic(ts, cfg, processed_scene_tokens, 
                               target_sample_token_override=None,
                               num_frames_override=num_frames_config)
    
    print(f"✓ Stage 1: GT Visualisierung abgeschlossen.")

def run_ground_truth_logic(ts, cfg, scenes_to_process_tokens, target_sample_token_override=None, num_frames_override=0):
    """
    Integrierte Logik aus examples/visualize_ground_truth.py.
    Wenn target_sample_token_override gesetzt ist, wird nur dieser Frame in der ersten Szene verarbeitet.
    Wenn num_frames_override > 0, werden so viele Frames pro Szene verarbeitet.
    """
    out_dir = cfg["output"]["ground_truth_dir"]
    os.makedirs(out_dir, exist_ok=True)

    sensor_chan = cfg["visualization"]["camera_channel"]
    thickness   = cfg["render"]["line_thickness"]
    z_thresh    = cfg["render"]["z_threshold"]
    box_vis     = cfg["render"]["box_visibility"]
    try:
        visibility  = BoxVisibility[box_vis.upper()]
    except KeyError:
        visibility = BoxVisibility.ANY
        print(f"WARNUNG: Ungültiger box_visibility Wert '{box_vis}'. Verwende 'ANY'.")
        
    gt_max      = cfg["dataset"].get("gt_max_boxes", None)

    for scene_token in scenes_to_process_tokens:
        if target_sample_token_override:
            sample_token = target_sample_token_override
            # Stelle sicher, dass der target_sample_token zur aktuellen Szene gehört (optionaler Check)
            if ts.get("sample", sample_token)["scene_token"] != scene_token and len(scenes_to_process_tokens) == 1:
                 # Wenn wir nur eine Szene (die des target_sample_tokens) verarbeiten, ist alles ok.
                 # Wenn wir über mehrere Szenen iterieren, aber einen globalen target_sample_token haben,
                 # müssten wir hier prüfen und ggf. nur für die passende Szene rendern.
                 # Für den Fall target_sample_idx >=0 wird nur eine Szene übergeben.
                 pass
            elif ts.get("sample", sample_token)["scene_token"] != scene_token:
                continue # Target sample gehört nicht zu dieser Szene in der Schleife

            effective_num_frames = 1 # Nur diesen einen Frame
        else:
            sample_token = ts.get("scene", scene_token)["first_sample_token"]
            effective_num_frames = num_frames_override if num_frames_override > 0 else float('inf')
        
        count = 0
        while sample_token and count < effective_num_frames:
            sample_record = ts.get("sample", sample_token)
            anns = sample_record["anns"]
            if gt_max is not None and gt_max > 0 : # gt_max muss auch > 0 sein
                anns = anns[:gt_max]

            out_path = os.path.join(out_dir, f"{scene_token}_{sample_token}.jpg")
            
            # Aufruf der render_sample_boxes Funktion (aus oft.utils.visualization)
            # Diese Funktion muss die Parameter visibility, z_threshold, thickness akzeptieren.
            try:
                render_sample_boxes(
                    ts,
                    sample_token=sample_token,
                    sensor_channel=sensor_chan,
                    visibility=visibility, # Wird übergeben
                    z_threshold=z_thresh,  # Wird übergeben
                    thickness=thickness,   # Wird übergeben
                    out_path=out_path
                )
                print(f"[GT] {scene_token}/{sample_token}: {len(anns)} Boxen → {out_path}")
            except TypeError as e:
                print(f"FEHLER beim Aufruf von render_sample_boxes für {sample_token}: {e}")
                print("  Stelle sicher, dass 'oft.utils.visualization.render_sample_boxes' die Argumente 'visibility', 'z_threshold' und 'thickness' unterstützt.")
                print("  Alternativ kann die Logik von render_sample_boxes hier direkt implementiert werden, um draw_boxes_on_image aus sensor_utils zu nutzen.")
                # Fallback: Logik aus sensor_utils.render_sample_boxes hier nachbilden, falls der Import fehlschlägt oder Signatur nicht passt
                # Dies wäre der robustere Weg, wenn oft.utils.visualization.render_sample_boxes nicht die erwartete Signatur hat.
                # from oft.utils.sensor_utils import get_camera_intrinsic, get_sensor_extrinsic, draw_boxes_on_image as draw_on_img_sensor_util, CLASS_COLORS
                # sd_token_fallback = sample_record["data"].get(sensor_chan)
                # if sd_token_fallback:
                #     sd_record_fallback = ts.get("sample_data", sd_token_fallback)
                #     img_fn_fallback = sd_record_fallback["filename"]
                #     if not os.path.isabs(img_fn_fallback): img_fn_fallback = os.path.join(ts.dataroot, img_fn_fallback)
                #     img_fallback = cv2.imread(img_fn_fallback)
                #     if img_fallback is not None:
                #         calib_fb = ts.get("calibrated_sensor", sd_record_fallback["calibrated_sensor_token"])
                #         ego_fb = ts.get("ego_pose", sd_record_fallback["ego_pose_token"])
                #         K_fb = get_camera_intrinsic(calib_fb)
                #         H_fb = get_sensor_extrinsic(ego_fb, calib_fb)
                #         boxes_fb = [ts.get_box(ann) for ann in anns]
                #         for b in boxes_fb: b.name="ground_truth" # Für Farbe
                #         oc = CLASS_COLORS.copy(); CLASS_COLORS.clear(); CLASS_COLORS["ground_truth"]=(0,255,0)
                #         global _debug_printed_gt, _debug_printed_fused; _debug_printed_gt = False; _debug_printed_fused=False
                #         img_out_fb = draw_on_img_sensor_util(img_fallback, boxes_fb, K_fb, H_fb, thickness, z_thresh, "gt_stage1_fb")
                #         CLASS_COLORS.clear(); CLASS_COLORS.update(oc)
                #         cv2.imwrite(out_path, img_out_fb)
                #         print(f"[GT-FB] {scene_token}/{sample_token}: {len(boxes_fb)} Boxen → {out_path}")
                #     else: print(f"FEHLER: Konnte Bild nicht für Fallback laden: {img_fn_fallback}")
                # else: print(f"FEHLER: Keine Sample Data für Fallback.")
            
            if target_sample_token_override and sample_token == target_sample_token_override:
                break # Nur den einen spezifischen Frame verarbeiten
            sample_token = sample_record.get("next")
            count += 1
        
        if target_sample_token_override: # Wenn wir einen spezifischen Frame gesucht haben, sind wir hier fertig.
            break

if __name__=="__main__":
    # Importiere cv2 hier, falls es nur in der Fallback-Logik verwendet wird, die vielleicht nicht immer erreicht wird
    # import cv2 
    # Importiere globale Debug-Flags, wenn sie im Modul-Scope von sensor_utils sind
    # from oft.utils.sensor_utils import _debug_printed_gt, _debug_printed_fused
    main()
