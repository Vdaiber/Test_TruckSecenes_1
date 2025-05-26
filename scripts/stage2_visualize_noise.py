#!/usr/bin/env python3
import os
# import sys # sys.path Manipulation in Docker meist nicht nötig, wenn CWD /app ist
# import yaml # load_config wird aus oft.utils importiert
import cv2
import numpy as np
import copy # Für deepcopy
from PIL import Image 
import argparse 

from oft.data.dataset import TruckScenesDataset 
from oft.utils.sensor_utils import (
    get_camera_intrinsic,
    get_sensor_extrinsic,
    draw_boxes_on_image, 
    CLASS_COLORS,
    _debug_printed_gt, _debug_printed_fused # Importiere Debug-Flags
)
from truckscenes import TruckScenes 
from truckscenes.utils.data_classes import Box as DevBox
# from truckscenes.utils.geometry_utils import view_points # view_points wird von draw_boxes_on_image intern verwendet

from oft.utils.config import load_config 

def try_load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None: return img
    try:
        pil_img = Image.open(path)
        if pil_img.mode == 'RGBA' or pil_img.mode == 'P': pil_img = pil_img.convert('RGB')
        arr = np.array(pil_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        print(f"WARNUNG: Konnte Bild {path} nicht mit PIL laden: {e}")
        return None

# filter_visible_boxes war in deinem Example, aber draw_boxes_on_image (Canvas sensor_utils_py_v1)
# hat jetzt eine eigene, verbesserte Logik für Z-Threshold und Kanten-Clipping.
# Daher ist eine separate filter_visible_boxes hier nicht mehr zwingend nötig,
# es sei denn, du willst eine Vorab-Filterung der Box-Liste.
# Für jetzt lassen wir es weg, um Redundanz zu vermeiden.

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Visualize Noise Comparison")
    parser.add_argument("-c", "--pipeline", required=False, default="config/pipeline.yaml",
                        help="Pfad zur pipeline.yaml.")
    args = parser.parse_args()

    cfg       = load_config(args.pipeline)
    dcfg      = cfg["dataset"]
    vcfg      = cfg["visualization"]
    rcfg      = cfg["render"]
    ocfg      = cfg["output"]
    
    out_dir   = ocfg.get("noise_comparison_dir", "/output/stage2_noise_visualization")
    os.makedirs(out_dir, exist_ok=True)

    cam_ch    = vcfg["camera_channel"]
    sample_idx_to_visualize  = vcfg.get("sample_idx", 0) # Ziel-Sample für den Vergleich
    thickness = rcfg.get("line_thickness", 2)
    z_thresh_render = rcfg.get("z_threshold", 0.1)
    noise_std_for_comparison = float(dcfg.get("augment_noise_std", 0.0))


    print(f"INFO Stage 2: Visualisiere Noise Vergleich für Sample-Index {sample_idx_to_visualize}")
    print(f"  Verwendete Noise StdDev für Vergleich: {noise_std_for_comparison}")

    # Dataset für saubere GT (augment_noise_std = 0.0)
    ds_clean_params = dict(
        dataroot       = dcfg["dataroot"],
        version        = dcfg["version"].strip(),
        history_window = 0, 
        max_boxes      = int(dcfg.get("gt_max_boxes") or 0), # oder 50 als Default
        augment_noise_std=0.0 # Wichtig: saubere Daten laden
    )
    ds_clean = TruckScenesDataset(**ds_clean_params)

    if not (0 <= sample_idx_to_visualize < len(ds_clean.samples)):
        print(f"FEHLER Stage 2: sample_idx {sample_idx_to_visualize} ist außerhalb des Dataset-Bereichs.")
        return

    target_sample_token = ds_clean.samples[sample_idx_to_visualize]
    print(f"  Sample Token: {target_sample_token}")

    # Lade Bild- und Kalibrierungsdaten
    ts_api = TruckScenes(version=dcfg["version"].strip(), dataroot=dcfg["dataroot"])
    samp_record   = ts_api.get("sample", target_sample_token)
    sd_token = samp_record["data"].get(cam_ch)
    if not sd_token:
        print(f"FEHLER Stage 2: Sample Data Token für Kanal {cam_ch} im Sample {target_sample_token} nicht gefunden.")
        return
    sd_record     = ts_api.get("sample_data", sd_token)
    
    img_fn = sd_record["filename"]
    if not os.path.isabs(img_fn):
        img_fn = os.path.join(ts_api.dataroot, img_fn)

    img = try_load_image(img_fn)
    if img is None:
        print(f"FEHLER Stage 2: Bild {img_fn} konnte nicht geladen werden.")
        return
    print(f"  Bild geladen: {img_fn}")

    calib_record = ts_api.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    ego_record   = ts_api.get("ego_pose", sd_record["ego_pose_token"])
    K_matrix     = get_camera_intrinsic(calib_record)
    H_world_to_sensor = get_sensor_extrinsic(ego_record, calib_record)

    # Hole saubere GT-Boxen (als DevBox Objekte in Weltkoordinaten)
    # Diese kommen direkt vom ts_api und sind daher nicht von ds_clean.augment_noise_std beeinflusst.
    gt_boxes_clean_devbox = [ts_api.get_box(ann_tok) for ann_tok in samp_record["anns"]]
    print(f"  {len(gt_boxes_clean_devbox)} saubere GT-Boxen (DevBox) geladen.")

    # Erzeuge verrauschte Boxen manuell für den Vergleich
    noisy_boxes_devbox = []
    if noise_std_for_comparison > 0:
        for gt_box in gt_boxes_clean_devbox:
            noisy_center = gt_box.center + np.random.normal(0.0, noise_std_for_comparison, 3)
            nb = DevBox(
                center=noisy_center.tolist(), # Wichtig: tolist() für JSON-ähnliche Struktur
                size=gt_box.wlh,
                orientation=gt_box.orientation,
                name=gt_box.name, 
                token=gt_box.token
            )
            noisy_boxes_devbox.append(nb)
    else: # Kein Rauschen -> noisy_boxes sind Kopien der sauberen
        noisy_boxes_devbox = [copy.deepcopy(b) for b in gt_boxes_clean_devbox]
    print(f"  {len(noisy_boxes_devbox)} verrauschte Boxen (DevBox) für Visualisierung erstellt.")
    
    # Farbgebung vorbereiten
    original_class_colors = CLASS_COLORS.copy()
    
    for box in gt_boxes_clean_devbox: box.name = "gt_clean_viz" # Eindeutiger Name für Farbe
    CLASS_COLORS["gt_clean_viz"] = (0,255,0) # Grün
    
    for box in noisy_boxes_devbox: box.name = "gt_noisy_viz" # Eindeutiger Name für Farbe
    CLASS_COLORS["gt_noisy_viz"] = (0,0,255) # Rot

    # Reset debug flags für draw_boxes_on_image
    global _debug_printed_gt, _debug_printed_fused
    _debug_printed_gt = False # Wird für "gt_clean_viz" verwendet
    _debug_printed_fused = False # Wird für "gt_noisy_viz" verwendet (als zweiter Satz)

    print(f"  Zeichne saubere GT (grün) und verrauschte GT (rot)...")
    img_display = draw_boxes_on_image(
        image=img, 
        boxes=gt_boxes_clean_devbox, 
        camera_k_matrix=K_matrix, 
        world_to_sensor_transform=H_world_to_sensor, 
        line_thickness=thickness,
        z_threshold=z_thresh_render,
        box_type_for_debug="gt" # Für die erste Debug-Ausgabe
        )
    img_display = draw_boxes_on_image(
        image=img_display, 
        boxes=noisy_boxes_devbox, 
        camera_k_matrix=K_matrix, 
        world_to_sensor_transform=H_world_to_sensor, 
        line_thickness=thickness,
        z_threshold=z_thresh_render,
        box_type_for_debug="fused" # Damit Debug-Ausgabe für zweiten Satz kommt
        )

    CLASS_COLORS.clear()
    CLASS_COLORS.update(original_class_colors)

    output_filename = f"sample_{sample_idx_to_visualize:03d}_noise_comparison.jpg"
    full_output_path = os.path.join(out_dir, output_filename)
    cv2.imwrite(full_output_path, img_display)
    print(f"✓ Stage 2: Noise Vergleichsbild gespeichert unter {full_output_path}")

if __name__=="__main__":
    main()