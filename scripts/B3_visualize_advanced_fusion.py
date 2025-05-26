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
from oft.utils.visualization import CLASS_COLORS as DEFAULT_CLASS_COLORS, _EDGES, draw_boxes_on_image as draw_devkit_style_boxes_on_image

COLOR_GT_STAGE1_STYLE = (0, 255, 0)
COLOR_FUSED = (255, 0, 0)
COLOR_FUSED_MULTI_SENSOR = (255, 165, 0)

CLASS_COLORS = DEFAULT_CLASS_COLORS if DEFAULT_CLASS_COLORS else {}

def world_box_dict_to_devbox(box_dict: Dict[str, Any], default_name="fused") -> DevBox:
    center = box_dict.get("box_world", [0,0,0,0,0,0,0])[:3]
    size = box_dict.get("box_world", [0,0,0,0,0,0,0])[3:6] 
    yaw = box_dict.get("box_world", [0,0,0,0,0,0,0])[6]
    orientation = Quaternion(axis=[0, 0, 1], angle=yaw)
    
    name_to_use = default_name
    num_contrib = box_dict.get("num_fused_tracks", 0)
    contrib_sensors = box_dict.get("contributing_sensors", [])
    
    if "detection_name" in box_dict: 
        name_to_use = box_dict["detection_name"]
    elif num_contrib == 1 and contrib_sensors:
        name_to_use = f"fused_{contrib_sensors[0]}"
    elif num_contrib > 1:
        name_to_use = f"fused_{num_contrib}sensors"
    
    token = str(box_dict.get("track_id", "fused_track_" + str(np.random.randint(10000))))
    return DevBox(center=center, size=size, orientation=orientation, name=name_to_use, token=token)

def draw_world_boxes_custom_text(
    image: np.ndarray,
    world_boxes: List[DevBox], 
    K: np.ndarray, 
    world_to_sensor_transform: np.ndarray, 
    box_color: Tuple[int, int, int],
    thickness: int = 2,
    z_threshold: float = 0.1,
    text_fn: Optional[Callable[[DevBox, int], str]] = None 
):
    img_out = image.copy()
    for i, box in enumerate(world_boxes):
        corners_world = box.corners()
        corners_world_h = np.vstack((corners_world, np.ones((1, 8)))) 
        corners_sensor_h = world_to_sensor_transform @ corners_world_h
        corners_sensor = corners_sensor_h[:3, :] 
        pts_projected = view_points(corners_sensor, K, normalize=True)
        xs, ys, zs = pts_projected
        if np.all(zs <= z_threshold): continue
        
        for k_edge, j_edge in _EDGES:
            if zs[k_edge] > z_threshold and zs[j_edge] > z_threshold:
                p1 = (int(xs[k_edge]), int(ys[k_edge]))
                p2 = (int(xs[j_edge]), int(ys[j_edge]))
                cv2.line(img_out, p1, p2, box_color, thickness, cv2.LINE_AA)
        
        if text_fn:
            text = text_fn(box, i) 
            text_anchor_candidates = []
            for corner_idx in range(8):
                 if zs[corner_idx] > z_threshold and \
                    0 <= xs[corner_idx] < img_out.shape[1] and \
                    0 <= ys[corner_idx] < img_out.shape[0]:
                    text_anchor_candidates.append((int(xs[corner_idx]), int(ys[corner_idx])))
            if text_anchor_candidates:
                text_anchor = min(text_anchor_candidates, key=lambda p: p[1])
                cv2.putText(img_out, text, (text_anchor[0] + 3, text_anchor[1] - 7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1, cv2.LINE_AA)
    return img_out

def main():
    parser = argparse.ArgumentParser(description="B3: Visualize Advanced Fusion vs Filtered Ground Truth")
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
        print(f"FEHLER B3: Kamera-Kanal '{camera_channel}' nicht im Sample {target_sample_token} gefunden.")
        return
    
    sd_token = target_sample_record["data"][camera_channel]
    sd_record = ts.get("sample_data", sd_token)
    
    image_path = Path(ts.dataroot) / sd_record["filename"]
    base_image = cv2.imread(str(image_path)) # Lade das Originalbild
    if base_image is None:
        print(f"FEHLER B3: Konnte Bild nicht laden von: {image_path}")
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
        print(f"FEHLER B3 beim Aufruf von ts.get_sample_data für sd_token {sd_token}: {e}")
        gt_boxes_cam_filtered = []
    
    print(f"  {len(gt_boxes_cam_filtered)} Ground Truth Boxen (Stage1-Style, gefiltert, Kamera-Koord.) geladen.")

    fused_tracks_file = final_fused_tracks_dir / f"fused_tracks_advanced_target_{target_sample_token}.json"
    fused_devboxes_world: List[DevBox] = []
    fused_tracks_data_for_text = [] 
    if fused_tracks_file.is_file():
        with open(fused_tracks_file, 'r') as f:
            fused_tracks_data_from_file_for_text = json.load(f)
        for track_dict in fused_tracks_data_from_file_for_text:
            try:
                dev_box = world_box_dict_to_devbox(track_dict)
                fused_devboxes_world.append(dev_box)
            except Exception as e:
                print(f"WARNUNG B3: Fehler beim Konvertieren des fusionierten Tracks '{track_dict.get('unique_id_before_fusion', track_dict.get('track_id'))}' in DevBox: {e}")
        print(f"  {len(fused_devboxes_world)} fusionierte Tracks geladen und in Welt-DevBox konvertiert.")
    else:
        print(f"WARNUNG B3: Datei mit fusionierten Tracks nicht gefunden: {fused_tracks_file}")

    # --- Boxen auf Bild zeichnen ---
    # Erstelle eine Kopie des Originalbilds für die kombinierte Darstellung
    image_to_draw_on = base_image.copy()

    # 1. Zeichne die (Stage1-gefilterten) GT-Boxen
    if gt_boxes_cam_filtered:
        original_colors = CLASS_COLORS.copy()
        temp_gt_color_name = "gt_b3_viz_filtered"
        for box in gt_boxes_cam_filtered: 
            box.name = temp_gt_color_name 
        CLASS_COLORS.clear() 
        CLASS_COLORS[temp_gt_color_name] = (0, 255, 0) # Grün für GT (RGB)

        # draw_devkit_style_boxes_on_image zeichnet auf das übergebene Bild und gibt es zurück
        image_to_draw_on = draw_devkit_style_boxes_on_image(
            image_to_draw_on, # Zeichne auf die aktuelle Version des Bildes
            gt_boxes_cam_filtered, 
            K_matrix,
            z_threshold=render_cfg.get("z_threshold", 0.1), 
            thickness=render_cfg.get("line_thickness", 2)
        )
        CLASS_COLORS.clear(); CLASS_COLORS.update(original_colors)
    else:
        print("INFO B3: Keine GT-Boxen nach Filterung zum Zeichnen vorhanden.")

    # 2. Zeichne die fusionierten Boxen auf dasselbe Bild
    def get_fused_track_text(devbox_for_text: DevBox, track_idx_in_list: int):
        original_track_dict_for_text = None
        for trk_dict_search in fused_tracks_data_from_file_for_text:
            expected_token_from_dict = str(trk_dict_search.get("track_id", -1))
            devbox_token_base = devbox_for_text.token.replace("fused_track_", "")
            if devbox_token_base == expected_token_from_dict:
                original_track_dict_for_text = trk_dict_search
                break
        if original_track_dict_for_text:
            score_text = f"{original_track_dict_for_text.get('confidence_score',0.0):.2f}"
            num_contrib_text = f"N{original_track_dict_for_text.get('num_fused_tracks',1)}"
            name_text = str(devbox_for_text.name)
            return f"F_{name_text}_{score_text}_{num_contrib_text}"
        return f"F_{str(devbox_for_text.name)}"

    for i, fused_world_box_instance in enumerate(fused_devboxes_world):
        original_track_dict_for_color = None
        for trk_dict_search_color in fused_tracks_data_from_file_for_text:
            expected_token_color = str(trk_dict_search_color.get("track_id", -1))
            fused_box_token_base = fused_world_box_instance.token.replace("fused_track_", "")
            if fused_box_token_base == expected_token_color:
                original_track_dict_for_color = trk_dict_search_color
                break
        
        current_color = COLOR_FUSED 
        if original_track_dict_for_color and original_track_dict_for_color.get("num_fused_tracks", 1) > 1:
            current_color = COLOR_FUSED_MULTI_SENSOR 

        image_to_draw_on = draw_world_boxes_custom_text( 
            image_to_draw_on, [fused_world_box_instance], K_matrix, world_to_sensor_transform,
            box_color=current_color, 
            thickness=render_cfg.get("line_thickness", 2),
            z_threshold=render_cfg.get("z_threshold", 0.1),
            text_fn=lambda b, idx_lambda, current_box_obj_for_text=fused_world_box_instance: get_fused_track_text(current_box_obj_for_text, i)
        )

    # --- Bild speichern ---
    output_image_filename = f"B3_fused_vs_gt_{target_sample_token}_{camera_channel}.jpg"
    output_image_path = b3_output_dir / output_image_filename
    cv2.imwrite(str(output_image_path), image_to_draw_on) # Speichere das finale Bild
    print(f"INFO B3: Visualisierung gespeichert unter: {output_image_path}")
    print(f"✓ B3: Visualisierung abgeschlossen.")

if __name__ == "__main__":
    main()