#!/usr/bin/env python3
import argparse
import json
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional # Optional und Tuple hier hinzugefügt
from pyquaternion import Quaternion # Für die Umwandlung von Yaw in Quaternion

# Interne Projekt-Imports
from oft.utils.config import load_config
from oft.data.dataset import TruckScenesDataset # Um GT-Daten und Kamerainfos zu laden
from truckscenes import TruckScenes # Für den Zugriff auf ts.get_box und ts.get_sample_data
from truckscenes.utils.data_classes import Box as DevBox # Die Box-Klasse des Devkits
from truckscenes.utils.geometry_utils import view_points # Für Transformationen und Projektion
# Importiere _EDGES und CLASS_COLORS aus deiner visualization.py
# Stelle sicher, dass der Pfad korrekt ist und oft.utils.visualization existiert
from oft.utils.visualization import CLASS_COLORS as DEFAULT_CLASS_COLORS, _EDGES 

# Definieren wir Farben für die Visualisierung
COLOR_GT = (0, 255, 0)      # Grün für Ground Truth
COLOR_FUSED = (255, 0, 0)   # Blau für fusionierte Tracks (Standard)
COLOR_FUSED_MULTI_SENSOR = (255, 165, 0) # Orange für Tracks aus >1 Sensor (optional)

# Verwende die importierten CLASS_COLORS, falls vorhanden, sonst Fallback
CLASS_COLORS = DEFAULT_CLASS_COLORS if DEFAULT_CLASS_COLORS else {
    "vehicle.car": (0, 0, 255), # Rot für Auto als Beispiel
    "human.pedestrian.adult": (255, 0, 0), # Blau für Fußgänger
    "default_obj": (128, 128, 128), # Grau als Default
}


def world_box_dict_to_devbox(box_dict: Dict[str, Any], default_name="fused") -> DevBox:
    """
    Konvertiert ein Track-Dictionary (mit Weltkoordinaten) in ein DevBox-Objekt.
    """
    center = box_dict.get("box_world", [0,0,0,0,0,0,0])[:3]
    size = box_dict.get("box_world", [0,0,0,0,0,0,0])[3:6] # w, l, h
    yaw = box_dict.get("box_world", [0,0,0,0,0,0,0])[6]
    
    # Konvertiere Yaw in Quaternion
    orientation = Quaternion(axis=[0, 0, 1], angle=yaw)
    
    # Name für DevBox
    name_to_use = default_name
    num_contrib = box_dict.get("num_fused_tracks", 0)
    contrib_sensors = box_dict.get("contributing_sensors", [])
    
    if num_contrib == 1 and contrib_sensors:
        name_to_use = f"fused_{contrib_sensors[0]}"
    elif num_contrib > 1:
        name_to_use = f"fused_{num_contrib}sensors"
    
    final_name_for_devbox = box_dict.get("detection_name", name_to_use) 
    # Versuche, den 'detection_name' aus dem Track zu verwenden, wenn er existiert und num_fused_tracks = 1 ist.
    # Dies setzt voraus, dass B1 'detection_name' in seine Tracks schreibt oder B2 es für nicht-fusionierte Tracks beibehält.
    if num_contrib == 1 and "detection_name" in box_dict:
         final_name_for_devbox = box_dict["detection_name"]


    token = str(box_dict.get("track_id", "fused_track_" + str(np.random.randint(10000))))

    return DevBox(center=center, size=size, orientation=orientation, name=final_name_for_devbox, token=token)

def draw_world_boxes_on_image_custom(
    image: np.ndarray,
    world_boxes: List[DevBox], 
    K: np.ndarray, 
    world_to_sensor_transform: np.ndarray, 
    color_override: Optional[Tuple[int, int, int]] = None, 
    default_color: Tuple[int, int, int] = (128, 128, 128), 
    thickness: int = 2,
    z_threshold: float = 0.1,
    text_to_display_fn: Optional[callable] = None # callable hinzugefügt
):
    """
    Zeichnet 3D-Boxen (gegeben in Weltkoordinaten) auf ein Bild.
    Transformiert die Box-Ecken zuerst in den Kamera-Koordinatenraum.
    """
    img_out = image.copy()
    
    for i, box in enumerate(world_boxes):
        corners_world = box.corners()  # (3,8)

        corners_world_h = np.vstack((corners_world, np.ones((1, 8)))) 
        corners_sensor_h = world_to_sensor_transform @ corners_world_h
        corners_sensor = corners_sensor_h[:3, :] 

        pts_projected = view_points(corners_sensor, K, normalize=True)  # (3,8)
        xs, ys, zs = pts_projected

        if np.all(zs <= z_threshold): 
            continue

        current_box_color_bgr = default_color
        if color_override:
            current_box_color_bgr = color_override
        else:
            # Farbe aus DevKit-Colormap basierend auf box.name
            # Stelle sicher, dass box.name ein String ist
            box_name_str = str(box.name) if box.name is not None else "default_obj"
            rgb_from_map = CLASS_COLORS.get(box_name_str, CLASS_COLORS.get(box_name_str.split('.')[0], None)) # Versuche auch Basis-Kategorie
            if rgb_from_map: 
                current_box_color_bgr = (int(rgb_from_map[2]), int(rgb_from_map[1]), int(rgb_from_map[0]))
            elif box_name_str not in CLASS_COLORS: # Fallback, wenn Name nicht in Map
                 current_box_color_bgr = default_color


        
        for k_edge, j_edge in _EDGES:
            if zs[k_edge] > z_threshold and zs[j_edge] > z_threshold:
                p1 = (int(xs[k_edge]), int(ys[k_edge]))
                p2 = (int(xs[j_edge]), int(ys[j_edge]))
                cv2.line(img_out, p1, p2, current_box_color_bgr, thickness, cv2.LINE_AA)
        
        if text_to_display_fn:
            text = text_to_display_fn(box, i) 
            text_anchor_candidates = []
            for corner_idx in range(8):
                 if zs[corner_idx] > z_threshold and \
                    0 <= xs[corner_idx] < img_out.shape[1] and \
                    0 <= ys[corner_idx] < img_out.shape[0]:
                    text_anchor_candidates.append((int(xs[corner_idx]), int(ys[corner_idx])))
            
            if text_anchor_candidates:
                text_anchor = min(text_anchor_candidates, key=lambda p: p[1])
                cv2.putText(img_out, text, (text_anchor[0], text_anchor[1] - 7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_box_color_bgr, thickness, cv2.LINE_AA)
    return img_out

def main():
    parser = argparse.ArgumentParser(description="B3: Visualize Advanced Fusion vs Ground Truth")
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
    
    final_fused_tracks_dir_str = ocfg.get("final_fused_tracks_dir_B2", "output/final_fused_tracks")
    final_fused_tracks_dir = Path(final_fused_tracks_dir_str)

    b3_output_dir_str = ocfg.get("b3_visualization_dir", "output/stageB3_adv_fusion_vs_gt")
    b3_output_dir = Path(b3_output_dir_str)
    b3_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"INFO B3: Initialisiere TruckScenes für GT und Kameradaten...")
    ts = TruckScenes(version=str(dcfg["version"]).strip(), dataroot=str(dcfg["dataroot"]))
    
    if not (0 <= target_sample_idx < len(ts.sample)):
        print(f"FEHLER B3: visualization.sample_idx ({target_sample_idx}) ist außerhalb des gültigen Bereichs.")
        return
    target_sample_record = ts.sample[target_sample_idx]
    target_sample_token = target_sample_record["token"]
    print(f"INFO B3: Visualisiere für Sample-Token: {target_sample_token} (Index: {target_sample_idx})")

    if camera_channel not in target_sample_record["data"]:
        print(f"FEHLER B3: Kamera-Kanal '{camera_channel}' nicht im Sample {target_sample_token} gefunden. Verfügbare Kanäle: {list(target_sample_record['data'].keys())}")
        return
    
    sd_token = target_sample_record["data"][camera_channel]
    sd_record = ts.get("sample_data", sd_token)
    
    image_path = Path(ts.dataroot) / sd_record["filename"]
    if not image_path.is_file():
        print(f"FEHLER B3: Bilddatei nicht gefunden: {image_path}")
        return
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"FEHLER B3: Konnte Bild nicht laden von: {image_path}")
        return

    calibrated_sensor_record = ts.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    ego_pose_record = ts.get("ego_pose", sd_record["ego_pose_token"])
    
    K = np.array(calibrated_sensor_record["camera_intrinsic"])
    
    # Welt-zu-Sensor Transformation für die Kamera
    ego_from_world_rotation = Quaternion(ego_pose_record['rotation']).inverse
    ego_from_world_translation = -ego_from_world_rotation.rotate(np.array(ego_pose_record['translation']))
    world_to_ego_transform = ego_from_world_rotation.transformation_matrix
    world_to_ego_transform[:3, 3] = ego_from_world_translation

    sensor_from_ego_rotation = Quaternion(calibrated_sensor_record['rotation']).inverse
    sensor_from_ego_translation = -sensor_from_ego_rotation.rotate(np.array(calibrated_sensor_record['translation']))
    ego_to_sensor_transform = sensor_from_ego_rotation.transformation_matrix
    ego_to_sensor_transform[:3, 3] = sensor_from_ego_translation
    
    world_to_sensor_transform = ego_to_sensor_transform @ world_to_ego_transform

    # --- Ground Truth Boxen laden und vorbereiten ---
    gt_devboxes_world: List[DevBox] = []
    for ann_token in target_sample_record["anns"]:
        gt_box_world = ts.get_box(ann_token) 
        gt_devboxes_world.append(gt_box_world)
    print(f"  {len(gt_devboxes_world)} Ground Truth Boxen geladen.")

    # --- Fusionierte Tracks laden und vorbereiten ---
    fused_tracks_file = final_fused_tracks_dir / f"fused_tracks_advanced_target_{target_sample_token}.json"
    fused_devboxes_world: List[DevBox] = []
    fused_tracks_data_from_file_for_text = [] # Um die Original-Dicts für Text zu behalten
    if fused_tracks_file.is_file():
        with open(fused_tracks_file, 'r') as f:
            fused_tracks_data_from_file_for_text = json.load(f)
        for track_dict in fused_tracks_data_from_file_for_text:
            try:
                dev_box = world_box_dict_to_devbox(track_dict)
                fused_devboxes_world.append(dev_box)
            except Exception as e:
                print(f"WARNUNG B3: Fehler beim Konvertieren des fusionierten Tracks in DevBox: {e} - Track: {track_dict.get('track_id')}")
        print(f"  {len(fused_devboxes_world)} fusionierte Tracks geladen und in DevBox konvertiert.")
    else:
        print(f"WARNUNG B3: Datei mit fusionierten Tracks nicht gefunden: {fused_tracks_file}")

    # --- Boxen auf Bild zeichnen ---
    image_with_gt = draw_world_boxes_on_image_custom(
        image.copy(), 
        gt_devboxes_world,
        K,
        world_to_sensor_transform,
        color_override=COLOR_GT, 
        thickness=render_cfg.get("line_thickness", 2),
        z_threshold=render_cfg.get("z_threshold", 0.1),
        text_to_display_fn=lambda box, idx: f"GT_{box.name.split('.')[-1] if isinstance(box.name, str) else 'GT'}"
    )
    
    # Funktion, um Text für fusionierte Boxen zu generieren
    # Diese Funktion greift auf fused_tracks_data_from_file_for_text aus dem äußeren Scope zu.
    def get_fused_track_text(devbox: DevBox, track_idx_in_current_list: int):
        original_track_dict = None
        # Finde das passende Dictionary in der Liste der geladenen Tracks.
        # Der DevBox-Token wurde aus der track_id des Dictionaries erstellt.
        for trk_dict in fused_tracks_data_from_file_for_text:
            # Erstelle den erwarteten Token-String aus dem Dictionary zum Vergleich
            expected_token = str(trk_dict.get("track_id", ""))
            if devbox.token == expected_token or devbox.token == f"fused_track_{expected_token}":
                original_track_dict = trk_dict
                break
        
        if original_track_dict:
            score_text = f"{original_track_dict.get('confidence_score',0.0):.2f}"
            num_contrib_text = f"N{original_track_dict.get('num_fused_tracks',1)}"
            # Verwende den Namen aus der DevBox, der in world_box_dict_to_devbox generiert wurde
            name_text = str(devbox.name).split('_')[-1] if 'fused_' in str(devbox.name) else str(devbox.name)
            return f"F_{name_text}_{score_text}_{num_contrib_text}"
        return f"F_{str(devbox.name)}"


    image_with_all_boxes = image_with_gt.copy() 
    for i, fused_box in enumerate(fused_devboxes_world):
        original_track_dict = None
        for trk_dict in fused_tracks_data_from_file_for_text:
            expected_token = str(trk_dict.get("track_id", ""))
            if fused_box.token == expected_token or fused_box.token == f"fused_track_{expected_token}":
                original_track_dict = trk_dict
                break
        
        current_color = COLOR_FUSED 
        if original_track_dict and original_track_dict.get("num_fused_tracks", 1) > 1:
            current_color = COLOR_FUSED_MULTI_SENSOR 

        image_with_all_boxes = draw_world_boxes_on_image_custom(
            image_with_all_boxes, [fused_box], K, world_to_sensor_transform,
            color_override=current_color, 
            thickness=render_cfg.get("line_thickness", 2),
            z_threshold=render_cfg.get("z_threshold", 0.1),
            # Übergabe des Index 'i' an die Lambda-Funktion, damit get_fused_track_text
            # das korrekte Element aus fused_tracks_data_from_file_for_text verwenden kann, falls nötig.
            # Aber get_fused_track_text sucht jetzt selbstständig.
            text_to_display_fn=lambda b, idx_lambda, current_box_obj=fused_box: get_fused_track_text(current_box_obj, i)
        )

    # --- Bild speichern ---
    output_image_filename = f"B3_fused_vs_gt_{target_sample_token}_{camera_channel}.jpg"
    output_image_path = b3_output_dir / output_image_filename
    cv2.imwrite(str(output_image_path), image_with_all_boxes)
    print(f"INFO B3: Visualisierung gespeichert unter: {output_image_path}")
    print(f"✓ B3: Visualisierung abgeschlossen.")

if __name__ == "__main__":
    main()