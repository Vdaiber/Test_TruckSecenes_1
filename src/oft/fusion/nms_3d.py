from typing import List
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate 

DEBUG_NMS = True 

def box_to_polygon(box: np.ndarray) -> Polygon:
    if not isinstance(box, np.ndarray) or box.shape[0] < 7:
        if DEBUG_NMS: print(f"DEBUG NMS (box_to_polygon): Ungültige Box-Eingabe: {box}. Erzeuge leeres Polygon.")
        return Polygon() 
    x, y, _, box_w, box_l, _, yaw = box[:7] 
    dx = box_l / 2.0
    dy = box_w / 2.0
    corners = np.array([
        [ dx,  dy], [ dx, -dy], [-dx, -dy], [-dx,  dy]  
    ], dtype=np.float64) 
    cos_y = np.cos(yaw); sin_y = np.sin(yaw)
    rot_mat = np.array([[cos_y, -sin_y], [sin_y,  cos_y]], dtype=np.float64)
    corners_rot = corners.dot(rot_mat.T) 
    corners_translated = corners_rot + np.array([x, y], dtype=np.float64) 
    poly = Polygon(corners_translated)
    if DEBUG_NMS and (not poly.is_valid or poly.is_empty):
        # print(f"DEBUG NMS (box_to_polygon): Erzeugtes Polygon für Box {box[0]:.1f},{box[1]:.1f} W:{box[3]:.2f} L:{box[4]:.2f} Y:{box[6]:.2f} ist ungültig/leer. Fläche: {poly.area:.4f}")
        pass
    return poly

def bev_iou(box1_global_coords: np.ndarray, box2_global_coords: np.ndarray) -> float:
    poly1_global = box_to_polygon(box1_global_coords)
    poly2_global = box_to_polygon(box2_global_coords)

    if not poly1_global.is_valid or poly1_global.is_empty: return 0.0
    if not poly2_global.is_valid or poly2_global.is_empty: return 0.0
    
    offset_x = -poly1_global.centroid.x
    offset_y = -poly1_global.centroid.y
    poly1_local = translate(poly1_global, xoff=offset_x, yoff=offset_y)
    poly2_local = translate(poly2_global, xoff=offset_x, yoff=offset_y)

    if not poly1_local.is_valid or poly1_local.is_empty: return 0.0
    if not poly2_local.is_valid or poly2_local.is_empty: return 0.0

    poly1_area = poly1_local.area 
    poly2_area = poly2_local.area

    if poly1_area < 1e-6 or poly2_area < 1e-6: 
        if DEBUG_NMS: 
            print(f"      DEBUG NMS (bev_iou): Polygonfläche zu klein. B1_Glob({box1_global_coords[0]:.1f},{box1_global_coords[1]:.1f} W:{box1_global_coords[3]:.2f} L:{box1_global_coords[4]:.2f}) Fl:{poly1_area:.4f}")
            print(f"      DEBUG NMS (bev_iou): Polygonfläche zu klein. B2_Glob({box2_global_coords[0]:.1f},{box2_global_coords[1]:.1f} W:{box2_global_coords[3]:.2f} L:{box2_global_coords[4]:.2f}) Fl:{poly2_area:.4f}")
        return 0.0

    try:
        inter_area = poly1_local.intersection(poly2_local).area
        union_area = poly1_area + poly2_area - inter_area
    except Exception: return 0.0

    if union_area < 1e-6: return 0.0
        
    iou = inter_area / union_area
    
    # NEU: Detailliertes Logging, wenn IoU exakt 0 ist, aber Flächen nicht
    if DEBUG_NMS and iou == 0.0 and poly1_area > 1e-6 and poly2_area > 1e-6:
        print(f"      DEBUG NMS (bev_iou): IoU=0.0 trotz pos. Flächen!")
        print(f"        Box1_Glob: {np.round(box1_global_coords,2).tolist()}, Poly1Fläche: {poly1_area:.4f}")
        print(f"        Box2_Glob: {np.round(box2_global_coords,2).tolist()}, Poly2Fläche: {poly2_area:.4f}")
        print(f"        Intersect-Fläche: {inter_area:.4f}, Union-Fläche: {union_area:.4f}")
        # Optional: Eckpunkte der lokalen Polygone ausgeben
        # print(f"        Poly1_local_coords: {list(poly1_local.exterior.coords)}")
        # print(f"        Poly2_local_coords: {list(poly2_local.exterior.coords)}")


    return max(0.0, min(iou, 1.0))

def nms_bev_3d(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
    if DEBUG_NMS: print(f"\nDEBUG NMS: Starte nms_bev_3d mit {len(boxes)} Boxen, IoU-Schwelle: {iou_threshold}")
    if len(boxes) == 0: return []
    if not (isinstance(boxes, np.ndarray) and boxes.ndim == 2 and boxes.shape[1] >= 7 and 
            isinstance(scores, np.ndarray) and scores.ndim == 1 and scores.shape[0] == boxes.shape[0]):
        if DEBUG_NMS: print("DEBUG NMS WARNUNG: Ungültige Eingabe für 'boxes' oder 'scores'.")
        return []

    order = np.argsort(scores)[::-1] 
    keep_indices: List[int] = []
    suppressed_mask = np.zeros(len(boxes), dtype=bool)

    for i_loop_idx, current_box_original_idx in enumerate(order):
        if suppressed_mask[current_box_original_idx]: continue 
            
        keep_indices.append(current_box_original_idx)
        if DEBUG_NMS: 
            print(f"  DEBUG NMS Loop {i_loop_idx+1}/{len(order)}: Betrachte Box Idx {current_box_original_idx} (Score: {scores[current_box_original_idx]:.4f}) -> BEIBEHALTEN")
        
        for j_loop_idx in range(i_loop_idx + 1, len(order)):
            candidate_box_original_idx = order[j_loop_idx]
            if suppressed_mask[candidate_box_original_idx]: continue 
            
            iou = bev_iou(boxes[current_box_original_idx], boxes[candidate_box_original_idx])
            if DEBUG_NMS: 
                 iou_print_str = f"{iou:.4f}"
                 # Detail-Log für IoU=0 wird jetzt in bev_iou gemacht
                 print(f"    DEBUG NMS: IoU mit Kandidat-Box Idx {candidate_box_original_idx} (Score {scores[candidate_box_original_idx]:.2f}) = {iou_print_str}")
            
            if iou > iou_threshold:
                suppressed_mask[candidate_box_original_idx] = True
                if DEBUG_NMS: print(f"      DEBUG NMS: Kandidat-Box Idx {candidate_box_original_idx} wird UNTERDRÜCKT (IoU {iou:.4f} > Schwelle {iou_threshold}).")
                
    sorted_keep_indices = sorted(keep_indices)
    if DEBUG_NMS: print(f"DEBUG NMS: Finale beibehaltene Indizes (sortiert): {sorted_keep_indices}")
    return sorted_keep_indices

if __name__ == "__main__":
    # (Testfälle bleiben wie im Canvas nms_3d_py_v4)
    print("Starte NMS Test...")
    DEBUG_NMS = True 
    test_boxes = np.array([
        [0, 0, 0, 2, 4, 1.5, 0.0],      
        [0.5, 0, 0, 2, 4, 1.5, 0.0],    
        [5, 0, 0, 2, 3, 1.5, 0.0],      
        [5.2, 0, 0, 2, 3, 1.5, np.pi/4] 
    ], dtype=np.float32)
    test_scores = np.array([0.9, 0.8, 0.7, 0.75], dtype=np.float32)
    
    print("\nTestfall 1: iou_threshold = 0.1")
    keep1 = nms_bev_3d(test_boxes, test_scores, iou_threshold=0.1)
    print(f"  Beibehaltene Indizes: {keep1}")

    test_boxes_large_coords = np.array([
        [730000.0, 5300000.0, 0, 2, 4, 1.5, 0.0],    
        [730000.5, 5300000.0, 0, 2, 4, 1.5, 0.0],    
        [730005.0, 5300000.0, 0, 2, 3, 1.5, 0.0]     
    ])
    test_scores_large_coords = np.array([0.9,0.8,0.7])
    print("\nTestfall mit großen Koordinaten: iou_threshold = 0.1")
    keep_large = nms_bev_3d(test_boxes_large_coords, test_scores_large_coords, iou_threshold=0.1)
    print(f"  Beibehaltene Indizes (große Koord.): {keep_large}")

    print("\nDirekter Test von bev_iou mit großen Koordinaten:")
    iou_val = bev_iou(test_boxes_large_coords[0], test_boxes_large_coords[1])
    print(f"  IoU zwischen B0_large und B1_large = {iou_val:.4f} (sollte > 0.5 sein)")
    iou_val2 = bev_iou(test_boxes_large_coords[0], test_boxes_large_coords[2])
    print(f"  IoU zwischen B0_large und B2_large = {iou_val2:.4f} (sollte 0.0 sein)")
    
    print("\nNMS Test abgeschlossen.")