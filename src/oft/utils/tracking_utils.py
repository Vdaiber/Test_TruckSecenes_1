import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional

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
        return prev_positions 
    return prev_positions + velocities * dt

def _match_detections_to_tracks_hungarian(
    predicted_track_centers_xy: np.ndarray, 
    current_detection_centers_xy: np.ndarray, 
    max_distance: float
) -> List[Tuple[int, int, float]]: # Gibt jetzt auch die Kosten des Matches zurück
    """
    Interne Matching-Funktion.
    Gibt Liste von (track_idx, detection_idx, cost) Paaren zurück.
    """
    if predicted_track_centers_xy.shape[0] == 0 or current_detection_centers_xy.shape[0] == 0:
        return []

    diff = predicted_track_centers_xy[:, None, :] - current_detection_centers_xy[None, :, :]
    cost_matrix = np.linalg.norm(diff, axis=-1)
    
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    
    matches = []
    for r, c in zip(row_idx, col_idx):
        cost = cost_matrix[r, c]
        if cost <= max_distance:
            matches.append((r, c, cost)) # Füge die Kosten zum Match-Tupel hinzu
    return matches


_NEXT_TRACK_ID = 0

class Track:
    """ Verwaltet den Zustand eines einzelnen verfolgten Objekts. """
    def __init__(self, initial_box_world: np.ndarray, initial_velocity_world: np.ndarray, 
                 track_id: int, initial_dt: float = 0.05, initial_score: float = 0.5): # Default-Startscore
        global _NEXT_TRACK_ID
        self.id = track_id
        self.box_world = initial_box_world 
        self.velocity_world = initial_velocity_world 
        self.age = 0  
        self.hits = 1 
        self.time_since_update = 0 
        self.dt_history = [initial_dt] 
        
        self.last_match_cost: Optional[float] = None # Kosten des letzten erfolgreichen Matches
        self.confidence_score: float = initial_score # Konfidenz des Tracks (0 bis 1)

    def predict(self, dt: float) -> np.ndarray:
        current_center = self.box_world[:3]
        if self.velocity_world is not None and self.velocity_world.shape[0] == 3:
            predicted_center = predict_future_positions(current_center.reshape(1,3), self.velocity_world.reshape(1,3), dt)
            return predicted_center.flatten()
        else:
            return current_center 

    def update(self, matched_box_world: np.ndarray, dt_for_this_update: float, 
               match_cost: float, max_match_distance_for_score_norm: float,
               new_velocity_world: Optional[np.ndarray] = None):
        
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

        self.last_match_cost = match_cost
        # Update confidence score: 1.0 für perfekten Match (cost=0), 0.0 für Match an max_distance
        # Kann über Zeit gefiltert/geglättet werden, hier einfache Neuberechnung.
        if max_match_distance_for_score_norm > 0:
            self.confidence_score = max(0.0, 1.0 - (match_cost / max_match_distance_for_score_norm))
        else: # Vermeide Division durch Null, falls max_distance 0 ist
            self.confidence_score = 1.0 if match_cost == 0 else 0.0
        
        # print(f"  TRACK {self.id} UPDATED: cost={match_cost:.2f}, new_score={self.confidence_score:.2f}, new_center_xy={self.box_world[:2].tolist()}")


class MultiObjectTracker:
    def __init__(self, max_age: int, min_hits_for_output: int, match_max_distance: float):
        global _NEXT_TRACK_ID
        _NEXT_TRACK_ID = 0 
        self.tracks: List[Track] = []
        self.max_age = max_age
        self.min_hits_for_output = min_hits_for_output
        self.match_max_distance = match_max_distance # Wird jetzt auch für Score-Normalisierung verwendet
        print(f"MultiObjectTracker initialisiert: max_age={max_age}, min_hits={min_hits_for_output}, max_dist={match_max_distance}")

    def _get_next_id(self) -> int:
        global _NEXT_TRACK_ID
        _NEXT_TRACK_ID += 1
        return _NEXT_TRACK_ID

    def update(self, current_detections_world: np.ndarray, 
                     current_velocities_world: Optional[np.ndarray], 
                     dt: float):
        
        predicted_track_centers_world_list = [] 
        track_indices_for_prediction = [] 
        for i, track in enumerate(self.tracks):
            predicted_center_xyz = track.predict(dt) 
            predicted_track_centers_world_list.append(predicted_center_xyz[:2]) 
            track_indices_for_prediction.append(i)
            track.time_since_update += dt 

        if not predicted_track_centers_world_list: 
            predicted_track_centers_world_xy_np = np.zeros((0,2))
        else:
            predicted_track_centers_world_xy_np = np.array(predicted_track_centers_world_list)
        
        # print(f"    DEBUG MOT.update: shape predicted_track_centers_world_xy_np: {predicted_track_centers_world_xy_np.shape}")

        if current_detections_world.shape[0] > 0:
            current_detection_centers_world_xy = current_detections_world[:, :2]
        else:
            current_detection_centers_world_xy = np.zeros((0,2))
        
        # print(f"    DEBUG MOT.update: shape current_detection_centers_world_xy: {current_detection_centers_world_xy.shape}")
            
        # `matches` enthält jetzt Tupel (track_pred_idx, det_idx, cost)
        matches_with_cost = _match_detections_to_tracks_hungarian(
            predicted_track_centers_world_xy_np,
            current_detection_centers_world_xy,
            self.match_max_distance # Verwende die Instanzvariable
        )
        
        matched_track_original_indices = set()
        matched_detection_indices = set()

        # Aktualisiere gematchte Tracks
        for pred_track_list_idx, det_idx, cost_of_match in matches_with_cost: # Entpacke auch die Kosten
            original_track_idx = track_indices_for_prediction[pred_track_list_idx] 
            track = self.tracks[original_track_idx]
            
            new_vel = current_velocities_world[det_idx] if current_velocities_world is not None and current_velocities_world.shape[0] > det_idx else None
            track.update(current_detections_world[det_idx], dt, 
                         match_cost=cost_of_match, # Übergebe die Kosten
                         max_match_distance_for_score_norm=self.match_max_distance) # Für Normalisierung
            
            matched_track_original_indices.add(original_track_idx)
            matched_detection_indices.add(det_idx)

        # Behandle ungematchte Tracks
        for i, track in enumerate(self.tracks):
            if i not in matched_track_original_indices:
                track.age += 1
                # Optional: Reduziere Konfidenz bei Nicht-Match
                # track.confidence_score *= 0.9 # Beispielhafte Reduktion

        # Erstelle neue Tracks für ungematchte Detektionen
        for i in range(current_detections_world.shape[0]):
            if i not in matched_detection_indices:
                initial_vel = current_velocities_world[i] if current_velocities_world is not None and current_velocities_world.shape[0] > i else np.zeros(3)
                # Neue Tracks starten mit einer mittleren Konfidenz, die sich bei Matches verbessert
                new_track = Track(current_detections_world[i], initial_vel, 
                                  self._get_next_id(), initial_dt=dt, initial_score=0.5) 
                self.tracks.append(new_track)

        # Entferne alte Tracks und bereite Ausgabe vor
        active_tracks_output = []
        surviving_tracks = []
        for track in self.tracks:
            if track.age <= self.max_age:
                surviving_tracks.append(track)
                if track.hits >= self.min_hits_for_output and track.age == 0 : # age == 0 -> wurde in diesem Frame geupdated/gesehen
                     active_tracks_output.append({
                        "track_id": track.id,
                        "box_world": track.box_world.tolist(), 
                        "velocity_world": track.velocity_world.tolist() if track.velocity_world is not None else [0,0,0],
                        "age": track.age,
                        "hits": track.hits,
                        "confidence_score": round(track.confidence_score, 4), # NEU
                        "last_match_cost": round(track.last_match_cost, 4) if track.last_match_cost is not None else None # NEU
                    })
        
        self.tracks = surviving_tracks
        return active_tracks_output