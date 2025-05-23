#!/usr/bin/env python3
import os
import cv2
import numpy as np
from pyquaternion import Quaternion
from truckscenes.utils.colormap import get_colormap
from truckscenes.utils.geometry_utils import view_points, transform_matrix 
from truckscenes.utils.data_classes import Box as DevBox 

CLASS_COLORS = get_colormap()
_debug_printed_gt = False
_debug_printed_fused = False

def get_camera_intrinsic(calib: dict) -> np.ndarray:
    intrinsic_data = calib.get("camera_intrinsic")
    if intrinsic_data is None:
        raise ValueError("Kameraintrinsik nicht in Kalibrierungsdaten gefunden.")
    K = np.array(intrinsic_data, dtype=np.float32).reshape(3, 3)
    return K

def get_sensor_extrinsic(ego_pose: dict, calib: dict) -> np.ndarray:
    q_e = Quaternion(ego_pose["rotation"])
    T_world_from_ego = transform_matrix(ego_pose["translation"], q_e, inverse=False)
    T_ego_from_world = np.linalg.inv(T_world_from_ego)

    q_s = Quaternion(calib["rotation"])
    T_sensor_from_ego = transform_matrix(calib["translation"], q_s, inverse=True) 

    H_world_to_sensor = T_sensor_from_ego @ T_ego_from_world
    return H_world_to_sensor


def draw_boxes_on_image(
    image: np.ndarray,
    boxes: list, 
    camera_k_matrix: np.ndarray, 
    world_to_sensor_transform: np.ndarray, 
    line_thickness: int = 2,
    z_threshold: float = 0.1,
    box_type_for_debug: str = "unknown"
) -> np.ndarray:
    global _debug_printed_gt, _debug_printed_fused
    
    img_height, img_width = image.shape[:2]
    img_out = image.copy()

    edges = [
        (0,1),(1,2),(2,3),(3,0), 
        (4,5),(5,6),(6,7),(7,4), 
        (0,4),(1,5),(2,6),(3,7)  
    ]
    
    local_debug_print_done = False
    if box_type_for_debug == "gt" and _debug_printed_gt:
        local_debug_print_done = True
    if box_type_for_debug == "fused" and _debug_printed_fused:
        local_debug_print_done = True

    for box_idx, box in enumerate(boxes):
        is_first_box_to_debug = (box_idx == 0 and not local_debug_print_done)

        if not isinstance(box, DevBox):
            if is_first_box_to_debug:
                print(f"    DEBUG DRAW ({box_type_for_debug}): Box {box_idx} ist kein DevBox Objekt, Typ: {type(box)}")
            continue

        # --- KORREKTUR HIER ---
        # DevBox.corners() liefert die Ecken bereits in Weltkoordinaten (3,8)
        box_corners_world = box.corners() 
        # --------------------

        if is_first_box_to_debug:
            print(f"    DEBUG DRAW ({box_type_for_debug}) Box {box_idx}: Welt-Zentrum (aus box.center)={box.center.tolist()}")
            print(f"    DEBUG DRAW ({box_type_for_debug}) Box {box_idx}: Ecken Welt (direkt von box.corners(), erste 2 Spalten):\n{box_corners_world[:,:2].T.tolist()}")
            # print(f"    DEBUG DRAW ({box_type_for_debug}) Box {box_idx}: K-Matrix=\n{camera_k_matrix}")
            # print(f"    DEBUG DRAW ({box_type_for_debug}) Box {box_idx}: H_world_to_sensor=\n{world_to_sensor_transform}")

        box_corners_world_h = np.vstack((box_corners_world, np.ones((1,8)))) 
        box_corners_sensor_h = world_to_sensor_transform @ box_corners_world_h 
        box_corners_sensor = box_corners_sensor_h[:3, :] 

        if is_first_box_to_debug:
            print(f"    DEBUG DRAW ({box_type_for_debug}) Box {box_idx}: Ecken Sensor (XYZ, erste 2 Spalten):\n{box_corners_sensor[:,:2].T.tolist()}")
            print(f"    DEBUG DRAW ({box_type_for_debug}) Box {box_idx}: Ecken Sensor Z-Werte: {box_corners_sensor[2, :].tolist()}")
            print(f"    DEBUG DRAW ({box_type_for_debug}) Box {box_idx}: z_threshold={z_threshold}")

        if np.all(box_corners_sensor[2, :] <= z_threshold):
            if is_first_box_to_debug:
                print(f"    DEBUG DRAW ({box_type_for_debug}) Box {box_idx}: VERWORFEN - Alle Ecken <= z_threshold.")
            if box_idx == 0: 
                if box_type_for_debug == "gt": _debug_printed_gt = True
                if box_type_for_debug == "fused": _debug_printed_fused = True
                local_debug_print_done = True
            continue

        image_points_raw = view_points(box_corners_sensor, camera_k_matrix, normalize=True)
        image_points = image_points_raw[:2,:].astype(int) 

        if is_first_box_to_debug:
            print(f"    DEBUG DRAW ({box_type_for_debug}) Box {box_idx}: Projizierte Bildpunkte (XY, erste 2 Ecken):\n{image_points[:,:2].T.tolist()}")
            print(f"    DEBUG DRAW ({box_type_for_debug}) Box {box_idx}: Bild-Dimensionen (H,W): {img_height}, {img_width}")

        box_color_bgr = CLASS_COLORS.get(getattr(box, 'name', "default_obj"), (0,0,255)) 
        if len(box_color_bgr) == 3: 
           box_color_bgr = (box_color_bgr[2], box_color_bgr[1], box_color_bgr[0])

        num_edges_drawn_for_this_box = 0
        for i,j in edges:
            p1_sensor_z = box_corners_sensor[2, i]
            p2_sensor_z = box_corners_sensor[2, j]

            if p1_sensor_z > z_threshold and p2_sensor_z > z_threshold:
                x1, y1 = image_points[0, i], image_points[1, i]
                x2, y2 = image_points[0, j], image_points[1, j]

                p1_in_image = (0 <= x1 < img_width) and (0 <= y1 < img_height)
                p2_in_image = (0 <= x2 < img_width) and (0 <= y2 < img_height)

                if p1_in_image or p2_in_image: 
                    cv2.line(img_out, (x1,y1), (x2,y2), box_color_bgr, line_thickness, cv2.LINE_AA)
                    num_edges_drawn_for_this_box +=1
        
        if is_first_box_to_debug and num_edges_drawn_for_this_box == 0 and not np.all(box_corners_sensor[2, :] <= z_threshold) :
             print(f"    DEBUG DRAW ({box_type_for_debug}) Box {box_idx}: KEINE KANTEN GEZEICHNET, obwohl Box nicht komplett hinter z_threshold.")
             for k_idx in range(8):
                 print(f"      Ecke {k_idx}: Sensor Z={box_corners_sensor[2,k_idx]:.2f}, Bild XY=({image_points[0,k_idx]}, {image_points[1,k_idx]})")

        if box_idx == 0: 
            if box_type_for_debug == "gt": _debug_printed_gt = True
            if box_type_for_debug == "fused": _debug_printed_fused = True
            local_debug_print_done = True
            
    return img_out

def render_sample_boxes(
    ts, sample_token: str, sensor_channel: str,
    thickness: int = 2, out_path: str = None
) -> np.ndarray:
    sample = ts.get("sample", sample_token)
    if sensor_channel not in sample["data"]:
        raise ValueError(f"Kanal '{sensor_channel}' nicht gefunden")
    
    sd_token = sample["data"][sensor_channel]
    sd = ts.get("sample_data", sd_token)
    
    fn = sd["filename"]
    if not os.path.isabs(fn):
        fn = os.path.join(ts.dataroot, fn)
    
    img = cv2.imread(fn, cv2.IMREAD_COLOR) 
    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {fn}")

    calib = ts.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego   = ts.get("ego_pose", sd["ego_pose_token"])
    K     = get_camera_intrinsic(calib)
    H_world_to_sensor = get_sensor_extrinsic(ego, calib)

    ann_tokens = sample["anns"]
    boxes = [ts.get_box(a) for a in ann_tokens]
    
    global _debug_printed_gt, _debug_printed_fused
    _debug_printed_gt = False 
    _debug_printed_fused = False 

    img_out = draw_boxes_on_image(img, boxes, K, H_world_to_sensor, thickness, box_type_for_debug="render_sample")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img_out)
    return img_out