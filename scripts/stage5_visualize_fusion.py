#!/usr/bin/env python3
import os
import json
import cv2 # Importiere cv2 direkt
import math
import numpy as np # Importiere numpy direkt
from pyquaternion import Quaternion
import argparse # <--- HIER IST DER FEHLENDE IMPORT

from oft.utils.config import load_config
from oft.data.dataset import TruckScenesDataset
from truckscenes import TruckScenes 
from truckscenes.utils.data_classes import Box as DevBox 
from oft.utils.sensor_utils import (
    get_camera_intrinsic, get_sensor_extrinsic, draw_boxes_on_image, CLASS_COLORS
)

def try_load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        from PIL import Image
        pil_img = Image.open(path)
        if pil_img.mode == 'RGBA' or pil_img.mode == 'P':
            pil_img = pil_img.convert('RGB')
        arr = np.array(pil_img)
        if arr.ndim==2: 
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) 
    except ImportError:
        print("WARNUNG: PIL nicht installiert, erweiterte Bildladefunktionen nicht verfügbar.")
        return None
    except Exception as e:
        print(f"FEHLER beim Laden des Bildes {path} mit PIL: {e}")
        return None

def main():
    # Argument Parser für die Konfigurationsdatei
    parser = argparse.ArgumentParser(description="Stage 5: Visualize Fusion Results vs Ground Truth")
    parser.add_argument("-c", "--pipeline", required=False, # Mache es optional, wenn load_config() einen Default hat
                        default="config/pipeline.yaml", # Standardpfad, falls -c nicht gegeben
                        help="Pfad zur pipeline.yaml. Standard: config/pipeline.yaml im CWD.")
    args = parser.parse_args()

    cfg   = load_config(args.pipeline) # Lade Config über den (ggf. default) Pfad

    dcfg  = cfg["dataset"]
    vcfg  = cfg["visualization"]
    ocfg  = cfg["output"]
    cam_ch = vcfg["camera_channel"]

    sample_idx_to_visualize = vcfg.get("sample_idx", 0)
    
    print(f"INFO Stage 5: Visualisiere Fusion für Sample-Index {sample_idx_to_visualize}")

    temp_ds_for_gt = TruckScenesDataset(
        dataroot       = dcfg["dataroot"],
        version        = dcfg["version"].strip(),
        history_window = 0, 
        max_boxes      = 0, 
        augment_noise_std = 0.0 
    )
    if sample_idx_to_visualize >= len(temp_ds_for_gt.samples):
        print(f"FEHLER: sample_idx {sample_idx_to_visualize} ist außerhalb des Dataset-Bereichs ({len(temp_ds_for_gt.samples)}).")
        return
    current_sample_token = temp_ds_for_gt.samples[sample_idx_to_visualize]
    print(f"  Sample Token für Index {sample_idx_to_visualize}: {current_sample_token}")

    ts     = TruckScenes(version=dcfg["version"].strip(), dataroot=dcfg["dataroot"])
    samp   = ts.get("sample", current_sample_token)
    sd_info = samp["data"].get(cam_ch) # Verwende .get() für sichereren Zugriff
    if not sd_info: # sd_info ist hier der Token oder None
        print(f"FEHLER: Sample Data Token für Kanal {cam_ch} im Sample {current_sample_token} nicht gefunden.")
        return
    sd = ts.get("sample_data", sd_info) # Lade das volle sample_data record mit dem Token

    img_fn = sd["filename"]
    if not os.path.isabs(img_fn):
        img_fn = os.path.join(str(dcfg["dataroot"]), img_fn) 
    
    print(f"  Lade Bild: {img_fn}")
    img    = try_load_image(img_fn)
    if img is None:
        print(f"FEHLER: Bild {img_fn} konnte nicht geladen werden.")
        return

    calib = ts.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego   = ts.get("ego_pose", sd["ego_pose_token"])
    K     = get_camera_intrinsic(calib) 
    H     = get_sensor_extrinsic(ego, calib) 

    gt_boxes = [ ts.get_box(ann_token) for ann_token in samp["anns"] ]
    print(f"  {len(gt_boxes)} saubere GT-Boxen geladen.")

    # Pfad zur Ausgabe von Stage 4 (fused_detections.json)
    fused_input_json_path = ocfg.get("fused_json", ocfg.get("dets_json")) # Nimm fused_json oder dets_json
    if not fused_input_json_path: # Fallback, falls keiner der Keys existiert
        fused_input_json_path = "/output/fused_detections.json"
        print(f"WARNUNG: Weder 'fused_json' noch 'dets_json' in output config gefunden. Verwende Fallback: {fused_input_json_path}")


    print(f"  Lade fusionierte Detektionen aus: {fused_input_json_path}")
    if not os.path.exists(fused_input_json_path):
        print(f"FEHLER: Fusions-Datei {fused_input_json_path} nicht gefunden. Bitte zuerst Stage 4 ausführen.")
        return
        
    with open(fused_input_json_path, "r") as f:
        fused_box_data_list_from_stage4 = json.load(f) 

    fus_boxes = []
    if not fused_box_data_list_from_stage4:
        print(f"  WARNUNG: Keine fusionierten Boxen in {fused_input_json_path} gefunden.")
    else:
        # fused_box_data_list_from_stage4 ist eine Liste von Box-Dictionaries
        # Alle sollten zum selben Sample-Token gehören (dem, der von Stage 4 verarbeitet wurde)
        # Wir filtern hier explizit nach dem current_sample_token, für den wir visualisieren wollen.
        num_total_fused_boxes = len(fused_box_data_list_from_stage4)
        
        for d in fused_box_data_list_from_stage4:
            if d.get("sample_token") == current_sample_token:
                world_translation = d["translation_world"]
                world_size_wlh = d["size_wlh"] 
                world_yaw = d["rotation_yaw_world"]
                
                fused_box_obj = DevBox(
                    center=world_translation, 
                    size=world_size_wlh,
                    orientation=Quaternion(axis=[0,0,1], angle=world_yaw), 
                    name="fused_object", # Name für Farbgebung
                    score=d.get("score_hits", 1.0) # Score aus den Hits
                )
                fus_boxes.append(fused_box_obj)
            # else:
                # print(f"  DEBUG: Überspringe fusionierte Box mit Token {d.get('sample_token')} (erwartet: {current_sample_token})")
        
        if num_total_fused_boxes > 0 and not fus_boxes:
             print(f"  WARNUNG: Die Datei {fused_input_json_path} enthält {num_total_fused_boxes} Boxen, "
                  f"aber keine für den aktuell zu visualisierenden Token ({current_sample_token}). "
                  "Stelle sicher, dass Stage 3 und 4 für denselben sample_idx gelaufen sind oder die Config von Stage 5 angepasst wird.")

        print(f"  {len(fus_boxes)} fusionierte Boxen für Visualisierung vorbereitet (für Sample {current_sample_token}).")

    original_class_colors = CLASS_COLORS.copy() 
    CLASS_COLORS.clear()
    CLASS_COLORS["ground_truth"] = (0,255,0) 
    CLASS_COLORS["fused_object"] = (0,0,255) 
    
    for gt_box in gt_boxes: 
        gt_box.name = "ground_truth"

    print(f"  Zeichne GT-Boxen (grün) und fusionierte Boxen (rot)...")
    img_with_gt = draw_boxes_on_image(img, gt_boxes, K, H, cfg.get("render",{}).get("line_thickness", 2))
    img_with_all_boxes = draw_boxes_on_image(img_with_gt, fus_boxes, K, H, cfg.get("render",{}).get("line_thickness", 2))

    CLASS_COLORS.clear()
    CLASS_COLORS.update(original_class_colors)

    output_comparison_dir = ocfg.get("fusion_comparison_dir", "/output/stage5_fusion_comparison")
    os.makedirs(output_comparison_dir, exist_ok=True)
    
    output_filename = f"{current_sample_token}_fusion_vs_gt.jpg"
    full_output_path = os.path.join(output_comparison_dir, output_filename)
    
    cv2.imwrite(full_output_path, img_with_all_boxes)
    print(f"✓ Stage 5: Visualisierung gespeichert unter {full_output_path}")

if __name__ == "__main__":
    main()