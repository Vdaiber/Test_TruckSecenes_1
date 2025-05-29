#!/usr/bin/env python3
import os
import json
import cv2 
import math
import numpy as np 
from pyquaternion import Quaternion
import argparse 

from oft.utils.config import load_config
from oft.data.old_dataset import TruckScenesDataset
from truckscenes import TruckScenes 
from truckscenes.utils.data_classes import Box as DevBox 
from oft.utils.sensor_utils import (
    get_camera_intrinsic, get_sensor_extrinsic, draw_boxes_on_image, CLASS_COLORS,
    _debug_printed_gt, _debug_printed_fused # Importiere die globalen Debug-Flags
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
    parser = argparse.ArgumentParser(description="Stage 5: Visualize Fused Simulated Detections vs Ground Truth")
    parser.add_argument("-c", "--pipeline", required=False, 
                        default="config/pipeline.yaml", 
                        help="Pfad zur pipeline.yaml. Standard: config/pipeline.yaml im CWD.")
    args = parser.parse_args()

    cfg   = load_config(args.pipeline) 

    dcfg  = cfg["dataset"]
    vcfg  = cfg["visualization"]
    ocfg  = cfg["output"]
    render_cfg = cfg.get("render", {}) 
    cam_ch = vcfg["camera_channel"]

    sample_idx_to_visualize = vcfg.get("sample_idx", 0)
    
    print(f"INFO Stage 5: Visualisiere Fusion für Sample-Index {sample_idx_to_visualize}")

    temp_ds_for_gt = TruckScenesDataset(
        dataroot       = dcfg["dataroot"],
        version        = dcfg["version"].strip(),
        history_window = 0, 
        max_boxes      = int(dcfg.get("gt_max_boxes") or 0), 
        augment_noise_std = 0.0 
    )
    if sample_idx_to_visualize >= len(temp_ds_for_gt.samples):
        print(f"FEHLER: sample_idx {sample_idx_to_visualize} ist außerhalb des Dataset-Bereichs ({len(temp_ds_for_gt.samples)}).")
        return
    current_sample_token = temp_ds_for_gt.samples[sample_idx_to_visualize]
    print(f"  Sample Token für Index {sample_idx_to_visualize}: {current_sample_token}")

    ts     = TruckScenes(version=dcfg["version"].strip(), dataroot=dcfg["dataroot"])
    samp   = ts.get("sample", current_sample_token)
    sd_info_token = samp["data"].get(cam_ch) 
    if not sd_info_token: 
        print(f"FEHLER: Sample Data Token für Kanal {cam_ch} im Sample {current_sample_token} nicht gefunden.")
        return
    sd = ts.get("sample_data", sd_info_token) 

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
    H_world_to_sensor = get_sensor_extrinsic(ego, calib) 

    gt_boxes = [ ts.get_box(ann_token) for ann_token in samp["anns"] ]
    print(f"  {len(gt_boxes)} saubere GT-Boxen geladen.")

    # Lade Fusions-Detektionen aus Stage 4 (jetzt die fusionierten simulierten Daten)
    fused_input_json_path = ocfg.get("fused_simulated_json", "/output/fused_simulated_detections.json") 
    
    print(f"  Lade fusionierte simulierte Detektionen aus: {fused_input_json_path}")
    if not os.path.exists(fused_input_json_path):
        print(f"FEHLER: Fusions-Datei {fused_input_json_path} nicht gefunden. Bitte zuerst Stage 4 (mit simulierten Daten) ausführen.")
        return
        
    with open(fused_input_json_path, "r") as f:
        fused_box_data_list_from_stage4 = json.load(f) # Dies ist eine flache Liste von Box-Dictionaries

    fus_boxes = [] 
    if not fused_box_data_list_from_stage4:
        print(f"  WARNUNG: Keine fusionierten Boxen in {fused_input_json_path} gefunden.")
    else:
        num_total_fused_boxes_in_file = len(fused_box_data_list_from_stage4)
        for d in fused_box_data_list_from_stage4:
            if d.get("sample_token") == current_sample_token: # Sicherstellen, dass es der richtige Frame ist
                world_translation = d.get("translation_world", [0,0,0]) 
                world_size_wlh = d.get("size_wlh", [1,1,1]) 
                world_yaw = d.get("rotation_yaw_world", 0.0) 
                
                fused_box_obj = DevBox(
                    center=world_translation, 
                    size=world_size_wlh, 
                    orientation=Quaternion(axis=[0,0,1], angle=world_yaw), 
                    name="fused_object", 
                    score=d.get("fusion_score", 1.0) # Score aus der Fusionsdatei
                )
                fus_boxes.append(fused_box_obj)
        
        if num_total_fused_boxes_in_file > 0 and not fus_boxes:
             print(f"  WARNUNG: Die Datei {fused_input_json_path} enthält {num_total_fused_boxes_in_file} Boxen, "
                  f"aber keine für den aktuell zu visualisierenden Token ({current_sample_token}).")

        print(f"  {len(fus_boxes)} fusionierte simulierte Boxen für Visualisierung vorbereitet (für Sample {current_sample_token}).")

    original_class_colors = CLASS_COLORS.copy() 
    CLASS_COLORS.clear()
    CLASS_COLORS["ground_truth"] = (0,255,0) 
    CLASS_COLORS["fused_object"] = (0,0,255) 
    
    for gt_box in gt_boxes: 
        gt_box.name = "ground_truth"

    # Reset debug flags
    global _debug_printed_gt, _debug_printed_fused
    _debug_printed_gt = False
    _debug_printed_fused = False

    print(f"  Zeichne GT-Boxen (grün) und fusionierte simulierte Boxen (blau)...")
    img_with_gt = draw_boxes_on_image(
        image=img, 
        boxes=gt_boxes, 
        camera_k_matrix=K, 
        world_to_sensor_transform=H_world_to_sensor, 
        line_thickness=render_cfg.get("line_thickness", 2),
        z_threshold=render_cfg.get("z_threshold", 0.1),
        box_type_for_debug="gt"
    )
    img_with_all_boxes = draw_boxes_on_image(
        image=img_with_gt, 
        boxes=fus_boxes, 
        camera_k_matrix=K, 
        world_to_sensor_transform=H_world_to_sensor, 
        line_thickness=render_cfg.get("line_thickness", 2),
        z_threshold=render_cfg.get("z_threshold", 0.1),
        box_type_for_debug="fused"
    )

    CLASS_COLORS.clear()
    CLASS_COLORS.update(original_class_colors)

    output_comparison_dir = ocfg.get("fusion_comparison_dir", "/output/stage5_fusion_comparison")
    os.makedirs(output_comparison_dir, exist_ok=True)
    
    output_filename = f"{current_sample_token}_sim_fusion_vs_gt.jpg" # Neuer Dateiname für Klarheit
    full_output_path = os.path.join(output_comparison_dir, output_filename)
    
    cv2.imwrite(full_output_path, img_with_all_boxes)
    print(f"✓ Stage 5: Visualisierung gespeichert unter {full_output_path}")

if __name__ == "__main__":
    main()