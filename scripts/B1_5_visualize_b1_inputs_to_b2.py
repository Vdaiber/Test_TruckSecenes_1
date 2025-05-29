#!/usr/bin/env python3
import argparse
import json
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
from pyquaternion import Quaternion

from oft.utils.config import load_config
from truckscenes import TruckScenes
from truckscenes.utils.data_classes import Box as DevBox
from truckscenes.utils.geometry_utils import view_points, BoxVisibility
# Korrigierter Import: Da B3_visualize_advanced_fusion.py im selben Ordner liegt,
# können wir es direkt importieren.
from B3_visualize_advanced_fusion import draw_world_boxes_custom_text

# Definiere Farben für die verschiedenen B1-Sensoren
B1_SENSOR_COLORS = {
    "sensor_gt_clean_dropout": (255, 0, 0),  # Blau
    "sensor_gt_noisy_A": (0, 0, 255),      # Rot
    "sensor_gt_noisy_B_dropout": (0, 255, 255), # Gelb
    "default": (128, 128, 128) # Grau für unbekannt
}
COLOR_GT = (0, 255, 0) # Grün für Ground Truth

def b1_track_to_devbox(track_dict: Dict[str, Any]) -> Optional[DevBox]:
    """ Konvertiert ein B1 Track Dictionary (aus der Zwischendatei) in ein DevBox Objekt. """
    box_world_list = track_dict.get("box_world")
    if not (isinstance(box_world_list, list) and len(box_world_list) == 7):
        print(f"WARNUNG (b1_track_to_devbox): Ungültige 'box_world' Daten für Track: {track_dict.get('unique_id_before_fusion')}")
        return None
    
    center = np.array(box_world_list[:3])
    size = np.array(box_world_list[3:6]) # W, L, H
    yaw = float(box_world_list[6])
    orientation = Quaternion(axis=[0, 0, 1], angle=yaw)
    
    name = track_dict.get("source_sensor", "unknown_sensor")
    token = str(track_dict.get("unique_id_before_fusion", "b1_track_" + str(np.random.randint(10000))))
    
    return DevBox(center=center, size=size, orientation=orientation, name=name, token=token)

def get_b1_track_text_fn(track_dict_for_text: Dict[str, Any]) -> str:
    """ Erstellt den Text String für eine B1 Track Box. """
    sensor = track_dict_for_text.get("source_sensor", "UNK")
    track_id_b1 = track_dict_for_text.get("original_track_id_in_source", track_dict_for_text.get("track_id", "N/A"))
    score = track_dict_for_text.get("confidence_score", 0.0)
    # Kürze den Sensornamen für die Anzeige
    sensor_short_name = sensor.replace('sensor_gt_', '').replace('_dropout', 'Drp').replace('_noisy_', 'N')
    return f"{sensor_short_name[:1]}_{track_id_b1}_S{score:.1f}"


def main():
    parser = argparse.ArgumentParser(description="B1.5: Visualize B1 Tracks (Input to B2 Grouping) vs GT")
    parser.add_argument("-c", "--pipeline", required=False, default="config/pipeline.yaml",
                        help="Pfad zur pipeline.yaml.")
    args = parser.parse_args()

    cfg = load_config(args.pipeline)
    dcfg = cfg["dataset"]
    vcfg = cfg["visualization"]
    ocfg = cfg["output"]
    render_cfg = cfg.get("render", {})

    target_sample_idx = int(vcfg.get("sample_idx", 0))
    camera_channel = vcfg.get("camera_channel", "CAMERA_FRONT") 
    
    b1_inputs_for_b2_dir_str = ocfg.get("b1_inputs_for_b2_dir", "output/intermediate_b1_tracks_for_b2_input")
    b1_inputs_for_b2_dir = Path(b1_inputs_for_b2_dir_str)

    b1_5_output_dir_str = ocfg.get("b1_5_visualization_dir", "output/stageB1_5_b1_inputs_viz") 
    b1_5_output_dir = Path(b1_5_output_dir_str)
    b1_5_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"INFO B1.5: Initialisiere TruckScenes für GT und Kameradaten...")
    ts = TruckScenes(version=str(dcfg["version"]).strip(), dataroot=str(dcfg["dataroot"]))
    
    if not (0 <= target_sample_idx < len(ts.sample)):
        print(f"FEHLER B1.5: visualization.sample_idx ({target_sample_idx}) ist außerhalb des gültigen Bereichs.")
        return
    target_sample_record = ts.sample[target_sample_idx]
    target_sample_token = target_sample_record["token"]
    print(f"INFO B1.5: Visualisiere B1-Tracks für Sample-Token: {target_sample_token} (Index: {target_sample_idx})")

    if camera_channel not in target_sample_record["data"]:
        print(f"FEHLER B1.5: Kamera-Kanal '{camera_channel}' nicht im Sample {target_sample_token} gefunden.")
        return
    
    sd_token = target_sample_record["data"][camera_channel]
    sd_record = ts.get("sample_data", sd_token)
    image_path = Path(ts.dataroot) / sd_record["filename"]
    base_image = cv2.imread(str(image_path))
    if base_image is None:
        print(f"FEHLER B1.5: Konnte Bild nicht laden von: {image_path}")
        return

    calibrated_sensor_record = ts.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    ego_pose_record = ts.get("ego_pose", sd_record["ego_pose_token"])
    K_matrix = np.array(calibrated_sensor_record["camera_intrinsic"])
    
    ego_from_world_rotation = Quaternion(ego_pose_record['rotation']).inverse
    ego_from_world_translation = -ego_from_world_rotation.rotate(np.array(ego_pose_record['translation']))
    world_to_ego_transform = ego_from_world_rotation.transformation_matrix
    world_to_ego_transform[:3, 3] = ego_from_world_translation
    sensor_from_ego_rotation = Quaternion(calibrated_sensor_record['rotation']).inverse
    sensor_from_ego_translation = -sensor_from_ego_rotation.rotate(np.array(calibrated_sensor_record['translation']))
    ego_to_sensor_transform = sensor_from_ego_rotation.transformation_matrix
    ego_to_sensor_transform[:3, 3] = sensor_from_ego_translation
    world_to_sensor_transform = ego_to_sensor_transform @ world_to_ego_transform

    box_vis_config = render_cfg.get("box_visibility", "ANY")
    try:
        visibility_filter = BoxVisibility[box_vis_config.upper()]
    except KeyError:
        visibility_filter = BoxVisibility.ANY
    
    try:
        _, gt_boxes_cam_filtered, _ = ts.get_sample_data(sd_token, box_vis_level=visibility_filter)
    except Exception as e:
        print(f"FEHLER B1.5 beim Aufruf von ts.get_sample_data für sd_token {sd_token}: {e}")
        gt_boxes_cam_filtered = []
    print(f"  {len(gt_boxes_cam_filtered)} Ground Truth Boxen (gefiltert, Kamera-Koord.) geladen.")

    b1_input_tracks_file = b1_inputs_for_b2_dir / f"b1_tracks_for_b2_input_target_{target_sample_token}.json"
    b1_devboxes_world: List[DevBox] = []
    b1_tracks_data_for_text: List[Dict[str,Any]] = []

    if b1_input_tracks_file.is_file():
        with open(b1_input_tracks_file, 'r') as f:
            b1_tracks_data_for_text = json.load(f) 
        for track_dict in b1_tracks_data_for_text:
            dev_box = b1_track_to_devbox(track_dict)
            if dev_box:
                b1_devboxes_world.append(dev_box)
        print(f"  {len(b1_devboxes_world)} B1-Input-Tracks geladen und in Welt-DevBox konvertiert aus: {b1_input_tracks_file}")
    else:
        print(f"WARNUNG B1.5: Datei mit B1-Input-Tracks nicht gefunden: {b1_input_tracks_file}")

    image_to_draw_on = base_image.copy()

    if gt_boxes_cam_filtered:
        def get_gt_text(devbox: DevBox, idx: int):
            return f"GT_{devbox.name.split('.')[0]}" 
        
        image_to_draw_on = draw_world_boxes_custom_text(
            image_to_draw_on, gt_boxes_cam_filtered, K_matrix, 
            np.eye(4), 
            box_color=COLOR_GT,
            thickness=render_cfg.get("line_thickness", 2),
            z_threshold=render_cfg.get("z_threshold", 0.1),
            text_fn=get_gt_text
        )

    # Zeichne die B1-Input-Tracks
    # Stelle sicher, dass b1_tracks_data_for_text die gleiche Länge hat wie b1_devboxes_world
    # oder greife auf das Dictionary über eine eindeutige ID zu, falls die Reihenfolge nicht garantiert ist.
    # Für diesen einfachen Fall nehmen wir an, die Reihenfolge bleibt nach der Konvertierung erhalten.
    
    # Erstelle eine Liste von Tupeln (DevBox, zugehöriges Original-Dictionary)
    # um sicherzustellen, dass die Textfunktion das richtige Dictionary verwendet.
    drawable_b1_tracks_with_data = []
    temp_b1_devboxes_world = [] # Zum Neuaufbau, falls einige Konvertierungen fehlschlagen

    if len(b1_devboxes_world) == len(b1_tracks_data_for_text): # Sollte der Fall sein
        for i in range(len(b1_devboxes_world)):
            drawable_b1_tracks_with_data.append(
                (b1_devboxes_world[i], b1_tracks_data_for_text[i])
            )
    else: # Fallback, falls Längen nicht übereinstimmen (sollte nicht passieren)
        print(f"WARNUNG B1.5: Längen-Mismatch zwischen DevBoxen ({len(b1_devboxes_world)}) und Track-Daten ({len(b1_tracks_data_for_text)}). Text könnte ungenau sein.")
        for dev_box in b1_devboxes_world: # Nur die DevBoxen verwenden, Text wird generisch
             drawable_b1_tracks_with_data.append((dev_box, {"source_sensor": dev_box.name, "track_id": dev_box.token}))


    for b1_world_box_instance, original_track_dict_for_drawing in drawable_b1_tracks_with_data:
        sensor_name = original_track_dict_for_drawing.get("source_sensor", "default")
        current_color = B1_SENSOR_COLORS.get(sensor_name, B1_SENSOR_COLORS["default"])
        
        # Erstelle eine Lambda-Funktion, die das korrekte Dictionary für den Textaufruf erfasst
        # Dies ist notwendig, da Lambdas in Schleifen sonst die letzte Variable binden könnten.
        def create_text_fn(track_data_dict):
            return lambda b, idx_lambda: get_b1_track_text_fn(track_data_dict)

        current_text_fn = create_text_fn(original_track_dict_for_drawing)

        image_to_draw_on = draw_world_boxes_custom_text( 
            image_to_draw_on, [b1_world_box_instance], K_matrix, world_to_sensor_transform,
            box_color=current_color, 
            thickness=render_cfg.get("line_thickness", 2),
            z_threshold=render_cfg.get("z_threshold", 0.1),
            text_fn=current_text_fn
        )

    output_image_filename = f"B1_5_inputs_to_b2_vs_gt_{target_sample_token}_{camera_channel}.jpg"
    output_image_path = b1_5_output_dir / output_image_filename
    cv2.imwrite(str(output_image_path), image_to_draw_on)
    print(f"INFO B1.5: Visualisierung gespeichert unter: {output_image_path}")
    print(f"✓ B1.5: Visualisierung der B1-Inputs abgeschlossen.")

if __name__ == "__main__":
    main()