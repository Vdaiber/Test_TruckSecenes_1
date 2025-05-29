#!/usr/bin/env python3
import os
import json
import cv2 
import numpy as np 
from pyquaternion import Quaternion
import argparse 
from typing import List, Optional, Dict, Any 

from oft.utils.config import load_config
from truckscenes import TruckScenes 
from truckscenes.utils.data_classes import Box as DevBox 
from truckscenes.utils.geometry_utils import BoxVisibility 
from oft.utils.sensor_utils import (
    get_camera_intrinsic, get_sensor_extrinsic, draw_boxes_on_image, CLASS_COLORS,
    _debug_printed_gt, _debug_printed_fused 
)

def try_load_image(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None: return img
    try:
        from PIL import Image
        pil_img = Image.open(path)
        if pil_img.mode == 'RGBA' or pil_img.mode == 'P': pil_img = pil_img.convert('RGB')
        arr = np.array(pil_img)
        if arr.ndim==2: return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) 
    except ImportError: return None
    except Exception: return None

def main():
    parser = argparse.ArgumentParser(description="Stage 5: Visualize Single Fused Box vs GT")
    parser.add_argument("-c", "--pipeline", required=False, 
                        default="config/pipeline.yaml", 
                        help="Pfad zur pipeline.yaml.")
    args = parser.parse_args()

    cfg   = load_config(args.pipeline) 
    dcfg  = cfg["dataset"]
    vcfg  = cfg["visualization"]
    ocfg  = cfg["output"]
    render_cfg = cfg.get("render", {}) 
    cam_ch = vcfg["camera_channel"]

    sample_idx_to_visualize = vcfg.get("sample_idx", 0)
    z_threshold_visualization = float(render_cfg.get("z_threshold", 0.0)) 
    print(f"INFO Stage 5: Visualisiere EINZELNE Fused Box für Sample-Index {sample_idx_to_visualize} mit z_threshold={z_threshold_visualization}")

    ts = TruckScenes(version=str(dcfg["version"]).strip(), dataroot=str(dcfg["dataroot"]))
    
    if not (0 <= sample_idx_to_visualize < len(ts.sample)):
        print(f"FEHLER: sample_idx {sample_idx_to_visualize} ist außerhalb des Dataset-Bereichs."); return
        
    current_sample_token = ts.sample[sample_idx_to_visualize]["token"]
    print(f"  Sample Token für Index {sample_idx_to_visualize}: {current_sample_token}")

    samp   = ts.get("sample", current_sample_token)
    sd_info_token = samp["data"].get(cam_ch) 
    if not sd_info_token: print(f"FEHLER: Sample Data Token für Kanal {cam_ch} nicht gefunden."); return
    sd = ts.get("sample_data", sd_info_token) 

    img_fn = sd["filename"]
    if not os.path.isabs(img_fn): img_fn = os.path.join(ts.dataroot, img_fn)
    
    img = try_load_image(img_fn)
    if img is None: print(f"FEHLER: Bild {img_fn} konnte nicht geladen werden."); return

    calib = ts.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego   = ts.get("ego_pose", sd["ego_pose_token"])
    K     = get_camera_intrinsic(calib) 
    H_world_to_sensor = get_sensor_extrinsic(ego, calib) 

    box_vis_level_str = render_cfg.get("box_visibility", "ANY") 
    try: visibility_filter = BoxVisibility[box_vis_level_str.upper()]
    except KeyError: visibility_filter = BoxVisibility.ANY; print(f"WARNUNG: Ungültiger box_visibility Wert. Verwende 'ANY'.")

    _, gt_boxes_cam_filtered_devbox, _ = ts.get_sample_data(sd_info_token, box_vis_level=visibility_filter)
    
    gt_boxes_world_devbox: List[DevBox] = []
    if gt_boxes_cam_filtered_devbox:
        for cam_box in gt_boxes_cam_filtered_devbox:
            world_box = ts.get_box(cam_box.token) 
            if world_box: world_box.name = "ground_truth"; gt_boxes_world_devbox.append(world_box)
    print(f"  {len(gt_boxes_world_devbox)} sichtbare GT-Boxen (Weltkoordinaten) geladen.")

    # Lade die neu benannte Fusions-Datei
    fused_input_json_path = ocfg.get("fused_single_box_per_gt_json", "/output/fused_single_box_per_gt.json") 
    
    print(f"  Lade 'single representation' fusionierte Detektionen aus: {fused_input_json_path}")
    if not os.path.exists(fused_input_json_path):
        print(f"FEHLER: Fusions-Datei {fused_input_json_path} nicht gefunden."); fused_box_data_list = [] 
    else:
        with open(fused_input_json_path, "r") as f: fused_box_data_list = json.load(f)

    fus_boxes_devbox = [] 
    if not fused_box_data_list:
        print(f"  WARNUNG: Keine Boxen in {fused_input_json_path} gefunden.")
    else:
        for d_idx, d in enumerate(fused_box_data_list):
            if d.get("sample_token") == current_sample_token: 
                fused_box_obj = DevBox(
                    center=d.get("translation_world", [0,0,0]), 
                    size=d.get("size_wlh", [1,1,1]), 
                    orientation=Quaternion(axis=[0,0,1], angle=d.get("rotation_yaw_world", 0.0)), 
                    name="fused_blue", # Einheitlicher Name für blaue Farbe
                    score=d.get("fusion_score", 1.0),
                    token=f"fused_repr_{current_sample_token}_{d_idx}" 
                )
                fus_boxes_devbox.append(fused_box_obj)
        print(f"  {len(fus_boxes_devbox)} 'single representation' fusionierte Boxen für Visualisierung vorbereitet.")

    original_class_colors = CLASS_COLORS.copy() 
    CLASS_COLORS.clear()
    CLASS_COLORS["ground_truth"] = (0,255,0) # Grün
    CLASS_COLORS["fused_blue"]   = (0,0,255) # Blau

    for gt_box in gt_boxes_world_devbox: gt_box.name = "ground_truth"
    
    global _debug_printed_gt, _debug_printed_fused 
    _debug_printed_gt = False; _debug_printed_fused = False

    img_with_gt = draw_boxes_on_image(img.copy(), gt_boxes_world_devbox, K, H_world_to_sensor, 
                                      render_cfg.get("line_thickness", 2), z_threshold_visualization, "gt")
    img_with_all_boxes = draw_boxes_on_image(img_with_gt, fus_boxes_devbox, K, H_world_to_sensor, 
                                             render_cfg.get("line_thickness", 2), z_threshold_visualization, "fused")

    CLASS_COLORS.clear(); CLASS_COLORS.update(original_class_colors) 

    output_dir = ocfg.get("fusion_comparison_dir", "/output/stage5_fusion_comparison")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{current_sample_token}_single_fused_vs_gt.jpg" # Neuer Name für Output
    full_output_path = os.path.join(output_dir, output_filename)
    
    cv2.imwrite(full_output_path, img_with_all_boxes)
    print(f"✓ Stage 5: Visualisierung gespeichert unter {full_output_path}")

if __name__ == "__main__":
    main()