#!/usr/bin/env python3
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional

# Behalten wir deine vorhandene Funktion
def predict_future_positions(
    prev_positions: np.ndarray, # (N,3) XYZ
    velocities: np.ndarray,     # (N,3)
    dt: float
) -> np.ndarray:
    """ Lineare Projektion im Welt-KS: p' = p + v·dt """
    if prev_positions.ndim == 1: 
        prev_positions = prev_positions.reshape(1, -1)
    if velocities.ndim == 1:
        velocities = velocities.reshape(1, -1)
    if prev_positions.shape[0] != velocities.shape[0]:
        # print(f"WARNUNG predict_future_positions: Shape-Mismatch pos({prev_positions.shape}) vs vel({velocities.shape}). Gebe prev_positions zurück.")
        return prev_positions 
    return prev_positions + velocities * dt

def _match_detections_to_tracks_hungarian(
    predicted_track_centers_xy: np.ndarray, # (N_tracks, 2) Welt XY
    current_detection_centers_xy: np.ndarray, # (N_dets, 2) Welt XY
    max_distance: float
) -> List[Tuple[int, int]]:
    """
    Interne Matching-Funktion.
    Nimmt prädizierte Track-Zentren und aktuelle Detektions-Zentren (alle Welt-XY).
    Gibt Liste von (track_idx, detection_idx) Paaren zurück.
    """
    if predicted_track_centers_xy.shape[0] == 0 or current_detection_centers_xy.shape[0] == 0:
        return []

    # Debug-Ausgabe für Shapes direkt vor der kritischen Operation
    print(f"    DEBUG _match_detections_to_tracks_hungarian: shape predicted_track_centers_xy: {predicted_track_centers_xy.shape}")
    print(f"    DEBUG _match_detections_to_tracks_hungarian: shape current_detection_centers_xy: {current_detection_centers_xy.shape}")
    print(f"    DEBUG _match_detections_to_tracks_hungarian: PRED XY (erste 2): {predicted_track_centers_xy[:2].tolist()}")
    print(f"    DEBUG _match_detections_to_tracks_hungarian: CURR XY (erste 2): {current_detection_centers_xy[:2].tolist()}")


    # Broadcasting:
    # predicted_track_centers_xy[:, None, :]  -> (N_tracks, 1, 2)
    # current_detection_centers_xy[None, :, :] -> (1, N_dets, 2)
    # Subtraktion ergibt (N_tracks, N_dets, 2)
    diff = predicted_track_centers_xy[:, None, :] - current_detection_centers_xy[None, :, :]
    print(f"    DEBUG _match_detections_to_tracks_hungarian: shape diff: {diff.shape}")

    cost_matrix = np.linalg.norm(diff, axis=-1)
    print(f"    DEBUG _match_detections_to_tracks_hungarian: shape cost_matrix: {cost_matrix.shape}")
    
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    
    matches = []
    for r, c in zip(row_idx, col_idx):
        cost = cost_matrix[r, c]
        if cost <= max_distance:
            matches.append((r, c))
    return matches


_NEXT_TRACK_ID = 0

class Track:
    def __init__(self, initial_box_world: np.ndarray, initial_velocity_world: np.ndarray, track_id: int, initial_dt: float = 0.05):
        global _NEXT_TRACK_ID
        self.id = track_id
        self.box_world = initial_box_world 
        self.velocity_world = initial_velocity_world 
        self.age = 0  
        self.hits = 1 
        self.time_since_update = 0 
        self.dt_history = [initial_dt] 

    def predict(self, dt: float) -> np.ndarray:
        current_center = self.box_world[:3]
        if self.velocity_world is not None and self.velocity_world.shape[0] == 3:
            # Sicherstellen, dass current_center (3,) und velocity_world (3,) sind
            predicted_center = predict_future_positions(current_center.reshape(1,3), self.velocity_world.reshape(1,3), dt)
            return predicted_center.flatten() # Zurück als (3,)
        else:
            return current_center 

    def update(self, matched_box_world: np.ndarray, dt_for_this_update: float, new_velocity_world: Optional[np.ndarray] = None):
        if new_velocity_world is not None:
            self.velocity_world = new_velocity_world
        elif dt_for_this_update > 1e-6 : 
             self.velocity_world = (matched_box_world[:3] - self.box_world[:3]) / dt_for_this_update
        
        self.box_world = matched_box_world 
        self.age = 0 
        self.hits += 1
        self.time_since_update = 0 
        self.dt_history.append(dt_for_this_update)
        if len(self.dt_history) > 5: self.dt_history.pop(0)


class MultiObjectTracker:
    def __init__(self, max_age: int, min_hits_for_output: int, match_max_distance: float):
        global _NEXT_TRACK_ID
        _NEXT_TRACK_ID = 0 
        self.tracks: List[Track] = []
        self.max_age = max_age
        self.min_hits_for_output = min_hits_for_output
        self.match_max_distance = match_max_distance
        print(f"MultiObjectTracker initialisiert: max_age={max_age}, min_hits={min_hits_for_output}, max_dist={match_max_distance}")

    def _get_next_id(self) -> int:
        global _NEXT_TRACK_ID
        _NEXT_TRACK_ID += 1
        return _NEXT_TRACK_ID

    def update(self, current_detections_world: np.ndarray, 
                     current_velocities_world: Optional[np.ndarray], 
                     dt: float):
        
        predicted_track_centers_world_list = [] # Geändert zu Python-Liste
        track_indices_for_prediction = [] 
        for i, track in enumerate(self.tracks):
            predicted_center_xyz = track.predict(dt) # Gibt (3,) zurück
            predicted_track_centers_world_list.append(predicted_center_xyz[:2]) # Nimm XY
            track_indices_for_prediction.append(i)
            track.time_since_update += dt 

        if not predicted_track_centers_world_list: 
            predicted_track_centers_world_xy_np = np.zeros((0,2))
        else:
            # Konvertiere die Liste von (2,)-Arrays in ein (N,2)-Array
            predicted_track_centers_world_xy_np = np.array(predicted_track_centers_world_list)
        
        # DEBUG: Shape genau hier prüfen
        print(f"    DEBUG MOT.update: shape predicted_track_centers_world_xy_np: {predicted_track_centers_world_xy_np.shape}")


        if current_detections_world.shape[0] > 0:
            current_detection_centers_world_xy = current_detections_world[:, :2]
        else:
            current_detection_centers_world_xy = np.zeros((0,2))
        
        # DEBUG: Shape genau hier prüfen
        print(f"    DEBUG MOT.update: shape current_detection_centers_world_xy: {current_detection_centers_world_xy.shape}")
            
        matches = _match_detections_to_tracks_hungarian(
            predicted_track_centers_world_xy_np,
            current_detection_centers_world_xy,
            self.match_max_distance
        )
        
        matched_track_original_indices = set()
        matched_detection_indices = set()

        for pred_track_list_idx, det_idx in matches:
            original_track_idx = track_indices_for_prediction[pred_track_list_idx] 
            track = self.tracks[original_track_idx]
            
            new_vel = current_velocities_world[det_idx] if current_velocities_world is not None and current_velocities_world.shape[0] > det_idx else None
            track.update(current_detections_world[det_idx], dt, new_velocity_world=new_vel)
            
            matched_track_original_indices.add(original_track_idx)
            matched_detection_indices.add(det_idx)

        for i, track in enumerate(self.tracks):
            if i not in matched_track_original_indices:
                track.age += 1

        for i in range(current_detections_world.shape[0]):
            if i not in matched_detection_indices:
                initial_vel = current_velocities_world[i] if current_velocities_world is not None and current_velocities_world.shape[0] > i else np.zeros(3)
                new_track = Track(current_detections_world[i], initial_vel, self._get_next_id(), initial_dt=dt)
                self.tracks.append(new_track)

        active_tracks_output = []
        surviving_tracks = []
        for track in self.tracks:
            if track.age <= self.max_age:
                surviving_tracks.append(track)
                if track.hits >= self.min_hits_for_output and track.age == 0 :
                     active_tracks_output.append({
                        "track_id": track.id,
                        "box_world": track.box_world.tolist(), 
                        "velocity_world": track.velocity_world.tolist() if track.velocity_world is not None else [0,0,0],
                        "age": track.age,
                        "hits": track.hits,
                    })
        
        self.tracks = surviving_tracks
        return active_tracks_output