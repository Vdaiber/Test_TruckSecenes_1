#!/usr/bin/env python3
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional, Any

# Importiere bev_iou aus dem fusion.nms_3d Modul
from oft.fusion.nms_3d import bev_iou

# Die globalen Konstanten W_DIST und W_IOU werden entfernt,
# da diese Werte nun aus der Konfiguration gelesen werden.

def predict_future_positions(
    prev_positions: np.ndarray, 
    velocities: np.ndarray,     
    dt: float
) -> np.ndarray:
    if prev_positions.ndim == 1: 
        prev_positions = prev_positions.reshape(1, -1)
    if velocities.ndim == 1:
        velocities = velocities.reshape(1, -1)
    if prev_positions.shape[0] != velocities.shape[0]:
        # Im Fehlerfall oder bei ungültigen Eingaben könnte hier eine robustere Behandlung erfolgen.
        # Für den Moment wird angenommen, dass die aufrufende Logik korrekte Shapes sicherstellt
        # oder dieser Fall nicht eintritt, wenn nur einzelne Tracks prädiziert werden.
        print(f"WARNUNG: Shape-Mismatch in predict_future_positions. Prev: {prev_positions.shape}, Vel: {velocities.shape}")
        return prev_positions # Fallback: Gib die vorherige Position zurück
    return prev_positions + velocities * dt

_NEXT_TRACK_ID = 0

class Track:
    def __init__(self, initial_box_world: np.ndarray, initial_velocity_world: np.ndarray, 
                 track_id: int, initial_dt: float = 0.05, initial_score: float = 0.5):
        global _NEXT_TRACK_ID
        self.id = track_id
        if not (isinstance(initial_box_world, np.ndarray) and initial_box_world.shape == (7,)):
            # Fallback, falls initial_box_world nicht korrekt formatiert ist
            print(f"FEHLER Track Init (ID {self.id}): initial_box_world ungültig (Shape: {initial_box_world.shape if isinstance(initial_box_world, np.ndarray) else type(initial_box_world)}). Setze auf Null-Box.")
            self.box_world: np.ndarray = np.zeros(7, dtype=np.float32)
        else:
            self.box_world: np.ndarray = initial_box_world.copy()

        if not (isinstance(initial_velocity_world, np.ndarray) and initial_velocity_world.shape == (3,)):
            self.velocity_world: np.ndarray = np.zeros(3, dtype=np.float32)
        else:
            self.velocity_world: np.ndarray = initial_velocity_world.copy()
            
        self.age: int = 0  
        self.hits: int = 1 
        self.time_since_update: float = 0.0
        self.dt_history: List[float] = [initial_dt] 
        self.last_match_cost: Optional[float] = None 
        self.confidence_score: float = initial_score

    def predict(self, dt: float) -> np.ndarray:
        predicted_box_world = self.box_world.copy()
        current_center = predicted_box_world[:3]
        if self.velocity_world is not None and self.velocity_world.shape == (3,):
            predicted_center_flat = predict_future_positions(
                current_center.reshape(1, 3), 
                self.velocity_world.reshape(1, 3), 
                dt
            ).flatten()
            predicted_box_world[:3] = predicted_center_flat
        return predicted_box_world

    def update(self, matched_box_world: np.ndarray, dt_for_this_update: float, 
               match_cost: float, # Die kombinierte Kosten des Matches
               # max_match_distance_for_score_norm wird nicht mehr direkt für Score-Berechnung hier benötigt,
               # da match_cost bereits eine kombinierte, grob normalisierte Metrik ist.
               # Wir behalten es in der Signatur, falls es später für andere Zwecke nützlich ist
               # oder um Kompatibilität mit älteren Aufrufen (falls vorhanden) nicht sofort zu brechen,
               # aber die Score-Berechnung wird angepasst.
               max_match_distance_for_score_norm: float, # Vorerst ungenutzt in der neuen Score-Berechnung
               new_velocity_world: Optional[np.ndarray] = None):
        
        if not (isinstance(matched_box_world, np.ndarray) and matched_box_world.shape == (7,)):
            print(f"WARNUNG Track Update (ID {self.id}): matched_box_world ungültig. Update wird übersprungen.")
            return

        previous_center_for_vel_calc = self.box_world[:3].copy()
        self.box_world = matched_box_world.copy() 
        
        if new_velocity_world is not None and new_velocity_world.shape == (3,):
            self.velocity_world = new_velocity_world.copy()
        elif dt_for_this_update > 1e-6 : 
             self.velocity_world = (self.box_world[:3] - previous_center_for_vel_calc) / dt_for_this_update
        
        self.age = 0 
        self.hits += 1
        self.time_since_update = 0 
        self.dt_history.append(dt_for_this_update)
        if len(self.dt_history) > 5: self.dt_history.pop(0)
        self.last_match_cost = match_cost
        
        # Anpassung der Score-Berechnung:
        # Da match_cost (kombinierte Kosten) jetzt grob im Bereich [0,1] liegt
        # (0 für perfekten Match, ~1 für schlechten Match an der Grenze),
        # können wir den Score direkter ableiten.
        self.confidence_score = max(0.0, 1.0 - match_cost)

def _match_detections_to_tracks_hungarian(
    predicted_track_boxes_7d: np.ndarray,    
    current_detection_boxes_7d: np.ndarray, 
    max_combined_cost: float,              
    # Parameter für die Kostenberechnung werden jetzt direkt übergeben:
    w_dist: float,
    w_iou: float,
    center_dist_norm_factor: float 
) -> List[Tuple[int, int, float]]:
    
    num_tracks = predicted_track_boxes_7d.shape[0]
    num_detections = current_detection_boxes_7d.shape[0]

    if num_tracks == 0 or num_detections == 0:
        return []

    cost_matrix = np.zeros((num_tracks, num_detections), dtype=np.float32)

    for i in range(num_tracks):
        for j in range(num_detections):
            track_box = predicted_track_boxes_7d[i]
            det_box = current_detection_boxes_7d[j]

            center_dist = np.linalg.norm(track_box[:2] - det_box[:2])
            norm_dist_cost = min(center_dist / center_dist_norm_factor, 1.0) if center_dist_norm_factor > 1e-6 else 1.0
            
            iou = bev_iou(track_box, det_box) # Importierte Funktion
            iou_cost = 1.0 - iou

            combined_cost = w_dist * norm_dist_cost + w_iou * iou_cost
            cost_matrix[i, j] = combined_cost
            
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    
    matches = []
    for r, c in zip(row_idx, col_idx):
        cost = cost_matrix[r, c]
        if cost <= max_combined_cost:
            matches.append((r, c, cost))
    return matches

class MultiObjectTracker:
    def __init__(self, tracker_config: Dict[str, Any]): # Erwartet jetzt einen Dict mit Tracking-Konfiguration
        global _NEXT_TRACK_ID
        _NEXT_TRACK_ID = 0 
        self.tracks: List[Track] = []

        # Lese Parameter aus dem übergebenen Konfigurations-Dictionary
        temporal_cfg = tracker_config.get("temporal", {})
        self.max_age: int = int(temporal_cfg.get("max_age", 3))
        self.min_hits_for_output: int = int(tracker_config.get("min_hits_to_report", 3))
        
        # Parameter für die neue kombinierte Kostenfunktion
        self.match_max_combined_cost: float = float(temporal_cfg.get("match_max_combined_cost", 0.7))
        self.w_dist: float = float(temporal_cfg.get("cost_dist_weight", 0.7))
        self.w_iou: float = float(temporal_cfg.get("cost_iou_weight", 0.3))
        self.center_dist_norm_factor: float = float(temporal_cfg.get("cost_dist_norm_factor", 15.0))

        print(f"MultiObjectTracker (mit IOU & Config) initialisiert:")
        print(f"  max_age={self.max_age}, min_hits={self.min_hits_for_output}")
        print(f"  match_max_combined_cost={self.match_max_combined_cost:.2f}")
        print(f"  cost_dist_weight={self.w_dist:.2f}, cost_iou_weight={self.w_iou:.2f}")
        print(f"  cost_dist_norm_factor={self.center_dist_norm_factor:.2f}")

    def _get_next_id(self) -> int:
        global _NEXT_TRACK_ID
        _NEXT_TRACK_ID += 1
        return _NEXT_TRACK_ID

    def update(self, current_detections_world_7d: np.ndarray, 
                     current_velocities_world: Optional[np.ndarray], 
                     dt: float):
        
        predicted_track_boxes_list = []
        track_indices_for_prediction = []
        for i, track_obj in enumerate(self.tracks):
            predicted_box_7d = track_obj.predict(dt)
            predicted_track_boxes_list.append(predicted_box_7d)
            track_indices_for_prediction.append(i)
            track_obj.time_since_update += dt

        if not predicted_track_boxes_list:
            predicted_track_boxes_np = np.zeros((0,7), dtype=np.float32)
        else:
            predicted_track_boxes_np = np.array(predicted_track_boxes_list, dtype=np.float32)

        if current_detections_world_7d.ndim == 1 and current_detections_world_7d.shape[0] == 0: # Leerer Input, aber nicht (0,7)
             current_detections_world_7d = np.zeros((0,7), dtype=np.float32)


        matches_with_cost = _match_detections_to_tracks_hungarian(
            predicted_track_boxes_np,
            current_detections_world_7d,
            max_combined_cost=self.match_max_combined_cost,
            w_dist=self.w_dist,
            w_iou=self.w_iou,
            center_dist_norm_factor=self.center_dist_norm_factor
        )
        
        matched_track_original_indices = set()
        matched_detection_indices = set()

        for pred_track_list_idx, det_idx, cost_of_match in matches_with_cost:
            original_track_idx = track_indices_for_prediction[pred_track_list_idx]
            track_to_update = self.tracks[original_track_idx]
            
            det_velocity = None
            if current_velocities_world is not None and current_velocities_world.shape[0] > det_idx:
                det_velocity = current_velocities_world[det_idx]
            
            track_to_update.update(
                current_detections_world_7d[det_idx], 
                dt, 
                match_cost=cost_of_match,
                max_match_distance_for_score_norm=1.0, # Platzhalter, da Score anders berechnet wird
                new_velocity_world=det_velocity
            )
            matched_track_original_indices.add(original_track_idx)
            matched_detection_indices.add(det_idx)

        for i, track_obj in enumerate(self.tracks):
            if i not in matched_track_original_indices:
                track_obj.age += 1

        for i in range(current_detections_world_7d.shape[0]):
            if i not in matched_detection_indices:
                initial_vel_new_track = np.zeros(3, dtype=np.float32)
                if current_velocities_world is not None and current_velocities_world.shape[0] > i:
                    initial_vel_new_track = current_velocities_world[i]
                
                new_track = Track(
                    current_detections_world_7d[i], 
                    initial_vel_new_track, 
                    self._get_next_id(), 
                    initial_dt=dt, 
                    initial_score=0.5
                )
                self.tracks.append(new_track)

        active_tracks_output = []
        surviving_tracks = []
        for track_obj in self.tracks:
            if track_obj.age <= self.max_age:
                surviving_tracks.append(track_obj)
                if track_obj.hits >= self.min_hits_for_output and track_obj.age == 0:
                     active_tracks_output.append({
                        "track_id": track_obj.id,
                        "box_world": track_obj.box_world.tolist(), 
                        "velocity_world": track_obj.velocity_world.tolist(),
                        "age": track_obj.age,
                        "hits": track_obj.hits,
                        "confidence_score": round(track_obj.confidence_score, 4),
                        "last_match_cost": round(track_obj.last_match_cost, 4) if track_obj.last_match_cost is not None else None
                    })
        self.tracks = surviving_tracks
        return active_tracks_output

# Die Testfunktion test_cost_matrix_logic() kann entfernt werden, wenn sie nicht mehr
# für das direkte Ausführen dieses Moduls benötigt wird, oder angepasst werden,
# um die Konfigurationsparameter an den Tracker zu übergeben.
# Für den Moment lasse ich sie aus, da der Fokus auf der Integration in B1 liegt.
