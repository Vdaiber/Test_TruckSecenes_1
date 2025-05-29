#!/usr/bin/env python3
import argparse
import json
import os # os Modul importieren
import cv2 # Falls noch nicht oben, für imread und imwrite
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable 
from pyquaternion import Quaternion

from oft.utils.config import load_config
from truckscenes import TruckScenes
from truckscenes.utils.data_classes import Box as DevBox
from truckscenes.utils.geometry_utils import view_points, BoxVisibility 
# NEUER IMPORT für die Sichtbarkeitsprüfung
from oft.utils.visibility_utils import is_world_box_visible_in_camera_view
from oft.utils.visualization import CLASS_COLORS # Importiere CLASS_COLORS

# Definiere Farben für die verschiedenen B1-Sensoren (ähnlich zu B1.5)
# Diese werden hier nicht direkt für die B3-Tracks verwendet, könnten aber als Referenz dienen.
B3_FUSED_COLORS = {
    "single_sensor_fused": (255, 0, 0),  # Blau für Fusion basierend auf einem B1-Track
    "multi_sensor_fused": (255, 165, 0), # Orange für Fusion aus mehreren B1-Tracks
    "default": (128, 128, 128)          # Grau für unbekannt
}
COLOR_GT_B3 = (0, 255, 0) # Grün für Ground Truth in B3

def world_box_dict_to_devbox(box_dict: Dict[str, Any], default_name="fused_adv") -> Optional[DevBox]:
    """ Konvertiert ein B2 Track Dictionary (aus der JSON-Datei) in ein DevBox Objekt. """
    box_world_list = box_dict.get("box_world")
    if not (isinstance(box_world_list, list) and len(box_world_list) == 7):
        # print(f"WARNUNG (world_box_dict_to_devbox): Ungültige 'box_world' Daten für Track: {box_dict.get('track_id')}")
        return None
    
    center = np.array(box_world_list[:3])
    size = np.array(box_world_list[3:6]) # W, L, H
    yaw = float(box_world_list[6])
    orientation = Quaternion(axis=[0, 0, 1], angle=yaw)
    
    # Name für die DevBox, um Farbe/Identifikation zu ermöglichen
    name_to_use = default_name
    num_contrib_sources = len(box_dict.get("contributing_sensors", []))
    if num_contrib_sources == 1:
        name_to_use = f"fused_single_src_{box_dict.get('contributing_sensors',[default_name])[0]}"
    elif num_contrib_sources > 1:
        name_to_use = f"fused_multi_src_{num_contrib_sources}"
    
    token = str(box_dict.get("track_id", "b2_track_" + str(np.random.randint(10000))))
    score = box_dict.get("confidence_score", 0.0) # Score mitnehmen
    
    return DevBox(center=center, size=size, orientation=orientation, name=name_to_use, token=token, score=score)

def draw_world_boxes_custom_text(
    image: np.ndarray,
    world_boxes: List[DevBox], 
    K: np.ndarray, 
    world_to_sensor_transform: np.ndarray, 
    box_color: Tuple[int, int, int], # Farbe wird jetzt direkt übergeben
    thickness: int = 2,
    z_threshold: float = 0.1,
    text_fn: Optional[Callable[[DevBox, int], str]] = None 
):
    img_out = image.copy()
    # Kanten einer 3D Box
    _EDGES = [
        (0,1),(1,2),(2,3),(3,0), # Untere Fläche
        (4,5),(5,6),(6,7),(7,4), # Obere Fläche
        (0,4),(1,5),(2,6),(3,7)  # Verbindungen unten-oben
    ]
    for i, box in enumerate(world_boxes):
        corners_world = box.corners() # (3,8) in Weltkoordinaten
        corners_world_h = np.vstack((corners_world, np.ones((1, 8)))) 
        corners_sensor_h = world_to_sensor_transform @ corners_world_h
        corners_sensor = corners_sensor_h[:3, :] 
        
        # Z-Threshold-Prüfung: Mindestens eine Ecke muss vor der Kamera und jenseits des Thresholds sein
        if not np.any(corners_sensor[2, :] > z_threshold):
            continue # Überspringe Box, wenn alle Ecken hinter oder zu nah sind

        pts_projected = view_points(corners_sensor, K, normalize=True) # (3,8) -> (x,y,depth_normalized)
        xs, ys, zs = pts_projected[0,:], pts_projected[1,:], corners_sensor[2,:] # Verwende echte Z-Werte für den Text

        # Zeichne Kanten
        for k_edge, j_edge in _EDGES:
            # Zeichne Kante nur, wenn BEIDE Punkte vor dem z_threshold liegen
            if zs[k_edge] > z_threshold and zs[j_edge] > z_threshold:
                p1 = (int(xs[k_edge]), int(ys[k_edge]))
                p2 = (int(xs[j_edge]), int(ys[j_edge]))
                # Stelle sicher, dass Punkte im Bild sind, um cv2 Fehler zu vermeiden
                # cv2.line kann außerhalb des Bildes zeichnen, aber es ist sauberer, es zu prüfen
                img_h, img_w = img_out.shape[:2]
                if (0 <= p1[0] < img_w and 0 <= p1[1] < img_h) or \
                   (0 <= p2[0] < img_w and 0 <= p2[1] < img_h):
                    cv2.line(img_out, p1, p2, box_color, thickness, cv2.LINE_AA)
        
        if text_fn:
            text_to_display = text_fn(box, i) 
            # Finde einen geeigneten Ankerpunkt für den Text (z.B. Ecke oben links, die im Bild ist)
            text_anchor_candidates = []
            for corner_idx in range(8): # Gehe alle 8 Ecken durch
                 if zs[corner_idx] > z_threshold and \
                    0 <= xs[corner_idx] < img_out.shape[1] and \
                    0 <= ys[corner_idx] < img_out.shape[0]:
                    text_anchor_candidates.append((int(xs[corner_idx]), int(ys[corner_idx])))
            
            if text_anchor_candidates:
                # Wähle den Punkt, der am höchsten und am weitesten links im Bild ist
                text_anchor = min(text_anchor_candidates, key=lambda p: (p[1], p[0])) 
                cv2.putText(img_out, text_to_display, (text_anchor[0] + 3, text_anchor[1] - 7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1, cv2.LINE_AA)
    return img_out

def main():
    parser = argparse.ArgumentParser(description="B3: Visualize Advanced Fusion vs Filtered Ground Truth")
    parser.add_argument("-c", "--pipeline", required=False, default="config/pipeline.yaml",
                        help="Pfad zur pipeline.yaml.")
    args = parser.parse_args()

    cfg = load_config(args.pipeline)
    dcfg = cfg["dataset"]
    vcfg = cfg["visualization"] # vcfg ist jetzt definiert
    ocfg = cfg["output"]
    render_cfg = cfg.get("render", {})

    target_sample_idx = int(vcfg.get("sample_idx", 0))
    camera_channel = vcfg.get("camera_channel", "CAMERA_FRONT") 
    
    final_fused_tracks_dir_str = ocfg.get("final_fused_tracks_dir_B2", "output/final_fused_tracks")
    final_fused_tracks_dir = Path(final_fused_tracks_dir_str)

    b3_output_dir_str = ocfg.get("b3_visualization_dir", "output/stageB3_adv_fusion_vs_gt")
    b3_output_dir = Path(b3_output_dir_str)
    b3_output_dir.mkdir(parents=True, exist_ok=True)

    ts = TruckScenes(version=str(dcfg["version"]).strip(), dataroot=str(dcfg["dataroot"]))
    
    if not (0 <= target_sample_idx < len(ts.sample)):
        print(f"FEHLER B3: visualization.sample_idx ({target_sample_idx}) ist außerhalb des gültigen Bereichs.")
        return
    target_sample_record = ts.sample[target_sample_idx]
    target_sample_token = target_sample_record["token"]
    print(f"INFO B3: Visualisiere für Sample-Token: {target_sample_token} (Index: {target_sample_idx})")

    if camera_channel not in target_sample_record["data"]:
        print(f"FEHLER B3: Kamera-Kanal '{camera_channel}' nicht im Sample {target_sample_token} gefunden.")
        return
    
    sd_token = target_sample_record["data"][camera_channel]
    sd_record = ts.get("sample_data", sd_token)
    
    image_path_str = sd_record["filename"]
    if not os.path.isabs(image_path_str):
        image_path_str = os.path.join(ts.dataroot, image_path_str)
    base_image = cv2.imread(image_path_str)
    if base_image is None:
        print(f"FEHLER B3: Konnte Bild nicht laden von: {image_path_str}")
        return
    image_height, image_width = base_image.shape[:2]


    calibrated_sensor_record = ts.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    ego_pose_record = ts.get("ego_pose", sd_record["ego_pose_token"])
    K_matrix = np.array(calibrated_sensor_record["camera_intrinsic"])
    
    # Welt-zu-Sensor Transformation Matrix
    # (Diese Logik ist aus oft.utils.sensor_utils.get_sensor_extrinsic())
    ego_from_world_rotation = Quaternion(ego_pose_record['rotation']).inverse
    ego_from_world_translation = -ego_from_world_rotation.rotate(np.array(ego_pose_record['translation']))
    world_to_ego_transform = ego_from_world_rotation.transformation_matrix
    world_to_ego_transform[:3, 3] = ego_from_world_translation
    
    sensor_from_ego_rotation = Quaternion(calibrated_sensor_record['rotation']).inverse
    sensor_from_ego_translation = -sensor_from_ego_rotation.rotate(np.array(calibrated_sensor_record['translation']))
    ego_to_sensor_transform = sensor_from_ego_rotation.transformation_matrix
    ego_to_sensor_transform[:3, 3] = sensor_from_ego_translation
    
    H_world_to_sensor = ego_to_sensor_transform @ world_to_ego_transform

    # Lade GT Boxen (gefiltert nach Sichtbarkeit durch get_sample_data)
    box_vis_config_gt = render_cfg.get("box_visibility", "ANY")
    try:
        visibility_filter_gt = BoxVisibility[box_vis_config_gt.upper()]
    except KeyError:
        visibility_filter_gt = BoxVisibility.ANY
        print(f"WARNUNG B3: Ungültiger box_visibility Wert '{box_vis_config_gt}'. Verwende 'ANY' für GT.")
    
    _, gt_boxes_cam_filtered_devbox, _ = ts.get_sample_data(sd_token, box_vis_level=visibility_filter_gt)
    
    gt_devboxes_world: List[DevBox] = []
    if gt_boxes_cam_filtered_devbox:
        for cam_box in gt_boxes_cam_filtered_devbox:
            world_box = ts.get_box(cam_box.token)
            if world_box:
                # Name für die GT Boxen setzen für die Farbgebung
                # (draw_world_boxes_custom_text verwendet die übergebene box_color)
                world_box.name = "ground_truth_b3" 
                gt_devboxes_world.append(world_box)
    print(f"  {len(gt_devboxes_world)} sichtbare Ground Truth Boxen (Weltkoordinaten) geladen.")

    # Lade fusionierte Tracks
    fused_tracks_file = final_fused_tracks_dir / f"fused_tracks_advanced_target_{target_sample_token}.json"
    all_fused_devboxes_world: List[DevBox] = []
    fused_tracks_data_for_text_map: Dict[str, Dict[str, Any]] = {} # Map von Box-Token zu Track-Dict

    if fused_tracks_file.is_file():
        with open(fused_tracks_file, 'r') as f:
            fused_tracks_data_from_file = json.load(f)
        for track_dict in fused_tracks_data_from_file:
            dev_box = world_box_dict_to_devbox(track_dict) 
            if dev_box:
                all_fused_devboxes_world.append(dev_box)
                fused_tracks_data_for_text_map[dev_box.token] = track_dict
        print(f"  {len(all_fused_devboxes_world)} fusionierte Tracks (Weltkoordinaten) geladen.")
    else:
        print(f"WARNUNG B3: Datei mit fusionierten Tracks nicht gefunden: {fused_tracks_file}")

    # Filtere fusionierte Boxen nach Sichtbarkeit
    visible_fused_devboxes_world: List[DevBox] = []
    visibility_filter_fused = BoxVisibility[render_cfg.get("box_visibility_fused", "ANY").upper()]
    
    print(f"  Filtere fusionierte Tracks nach Sichtbarkeit (Filter: {visibility_filter_fused})...")
    for world_box in all_fused_devboxes_world:
        if is_world_box_visible_in_camera_view(
            world_box, ts, sd_token, 
            image_width, image_height, 
            box_vis_level=visibility_filter_fused
        ):
            visible_fused_devboxes_world.append(world_box)
    print(f"  {len(visible_fused_devboxes_world)} von {len(all_fused_devboxes_world)} fusionierten Tracks als sichtbar eingestuft.")

    image_to_draw_on = base_image.copy()
    line_thickness_render = render_cfg.get("line_thickness", 2)
    z_threshold_render = float(render_cfg.get("z_threshold", 0.1)) 

    # Zeichne GT-Boxen
    if gt_devboxes_world:
        def get_gt_text_b3(devbox: DevBox, idx: int):
            # Extrahiere Kategorie-Namen, falls vorhanden und als Teil von box.name gespeichert
            cat_name = devbox.name.replace("ground_truth_b3", "").strip("_")
            return f"GT_{cat_name}" if cat_name else "GT"

        image_to_draw_on = draw_world_boxes_custom_text(
            image_to_draw_on, gt_devboxes_world, K_matrix, 
            H_world_to_sensor, 
            box_color=COLOR_GT_B3, # Explizit Grün
            thickness=line_thickness_render,
            z_threshold=z_threshold_render, 
            text_fn=get_gt_text_b3 
        )
        
    # Zeichne die gefilterten fusionierten Boxen
    for fused_world_box in visible_fused_devboxes_world:
        original_track_dict = fused_tracks_data_for_text_map.get(fused_world_box.token)
        if not original_track_dict: # Sollte nicht passieren, wenn Map korrekt gefüllt wurde
            original_track_dict = {"contributing_sensors": [fused_world_box.name], "track_id": "N/A", "confidence_score": fused_world_box.score if fused_world_box.score is not None else 0.0}

        num_contrib = len(original_track_dict.get("contributing_sensors", []))
        current_color = B3_FUSED_COLORS["default"]
        if num_contrib == 1:
            current_color = B3_FUSED_COLORS["single_sensor_fused"]
        elif num_contrib > 1:
            current_color = B3_FUSED_COLORS["multi_sensor_fused"]
        
        def get_b3_fused_text(track_data_dict_local: Dict[str, Any]) -> str:
            score_val = track_data_dict_local.get('confidence_score', 0.0)
            num_fused_val = track_data_dict_local.get('num_fused_tracks', len(track_data_dict_local.get("contributing_sensors",[]))) # Bessere Schätzung für num_fused
            track_id_val = track_data_dict_local.get('track_id', 'N/A')
            return f"F{num_fused_val}_{track_id_val}_S{score_val:.2f}"

        text_function = lambda b, idx, current_track_data=original_track_dict: get_b3_fused_text(current_track_data)

        image_to_draw_on = draw_world_boxes_custom_text( 
            image_to_draw_on, [fused_world_box], K_matrix, H_world_to_sensor,
            box_color=current_color, 
            thickness=line_thickness_render,
            z_threshold=z_threshold_render, 
            text_fn=text_function
        )

    output_image_filename = f"B3_fused_vs_gt_{target_sample_token}_{camera_channel}.jpg"
    output_image_path = b3_output_dir / output_image_filename
    cv2.imwrite(str(output_image_path), image_to_draw_on)
    print(f"INFO B3: Visualisierung gespeichert unter: {output_image_path}")
    print(f"✓ B3: Visualisierung abgeschlossen.")

if __name__ == "__main__":
    main()

