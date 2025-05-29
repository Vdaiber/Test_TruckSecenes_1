#!/usr/bin/env python3
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional

# Temporär hier für Testbarkeit, normalerweise: from oft.fusion.nms_3d import bev_iou
from shapely.geometry import Polygon # Erforderlich für bev_iou

def box_to_polygon(box: np.ndarray) -> Polygon:
    """
    Konvertiert eine 3D-Box (Center x,y,z, Länge, Breite, Höhe, Yaw) in ein 2D-Polygon (BEV).
    Erwartetes Format von box: [x, y, z, length, width, height, yaw] (7 Elemente)
    """
    if box.shape != (7,):
        raise ValueError(f"Box muss 7 Elemente haben (x,y,z,l,w,h,yaw), bekam aber Shape {box.shape}")
    x, y, _, length, width, _, yaw = box
    # Eckpunkte der Box vor Rotation (um z-Achse)
    # Shapely erwartet Ecken in einer bestimmten Reihenfolge für Polygonerstellung
    # (x +/- halbe Länge, y +/- halbe Breite)
    dx = length / 2.0
    dy = width / 2.0
    
    # Definiere Ecken relativ zum Box-Zentrum (0,0) im Box-Koordinatensystem
    # Reihenfolge: z.B. vorne-links, vorne-rechts, hinten-rechts, hinten-links
    corners_rel = np.array([
        [ dx,  dy], # Vorne-rechts (aus Sicht der Box-Orientierung)
        [-dx,  dy], # Hinten-rechts
        [-dx, -dy], # Hinten-links
        [ dx, -dy]  # Vorne-links
    ])
    
    # Rotationsmatrix für Yaw
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    rot_mat = np.array([[cos_y, -sin_y], 
                        [sin_y,  cos_y]])
    
    # Rotiere Ecken
    corners_rotated = corners_rel @ rot_mat.T # (4,2)
    
    # Translatiere Ecken zum Welt-Box-Zentrum
    corners_world = corners_rotated + np.array([x, y]) # Addiere x,y zu den rotierten Ecken
    
    return Polygon(corners_world)

def bev_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Berechnet die Intersection-over-Union (IoU) zweier 3D-Boxen in der Bodenebene (BEV).
    Boxen sind np.ndarray der Form (7,) [x,y,z,l,w,h,yaw].
    """
    try:
        poly1 = box_to_polygon(box1)
        poly2 = box_to_polygon(box2)
    except ValueError as e:
        print(f"Fehler bei Polygonerstellung für IOU: {e}")
        return 0.0

    if not poly1.is_valid or not poly2.is_valid:
        # print(f"Warnung: Ungültiges Polygon für IOU Berechnung. Box1: {box1.tolist()}, Box2: {box2.tolist()}")
        return 0.0 # Bei ungültigen Polygonen 0 IOU zurückgeben
        
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.area + poly2.area - intersection_area
    
    if union_area == 0:
        return 0.0 # Vermeide Division durch Null, wenn beide Flächen 0 sind oder sich exakt aufheben
    return intersection_area / union_area

# --- Ende der kopierten/angepassten bev_iou Logik ---


# Gewichte für die kombinierte Kostenfunktion
# Diese könnten später aus der Config geladen werden
W_DIST = 0.7  # Gewicht für die Distanzkosten
W_IOU  = 0.3  # Gewicht für die IOU-Kosten (1 - IOU)

def _match_detections_to_tracks_hungarian(
    predicted_track_boxes_7d: np.ndarray,      # Erwartet jetzt (M, 7)
    current_detection_boxes_7d: np.ndarray,  # Erwartet jetzt (N, 7)
    max_combined_cost: float,                # Max. erlaubte kombinierte Kosten
    max_center_dist_for_norm: float = 10.0   # Distanz zur Normalisierung (größer als typische max_distance)
) -> List[Tuple[int, int, float]]:
    """
    Interne Matching-Funktion mit kombinierter Kostenmetrik (Distanz + IOU).
    Gibt Liste von (track_idx, detection_idx, combined_cost) Paaren zurück.

    Args:
        predicted_track_boxes_7d: Array der prädizierten Track-Boxen (M, 7).
        current_detection_boxes_7d: Array der aktuellen Detektions-Boxen (N, 7).
        max_combined_cost: Schwellenwert für die maximal akzeptierten kombinierten Kosten.
        max_center_dist_for_norm: Ein Wert, um die Zentrum-Distanz zu normalisieren.
                                  Sollte größer sein als die typische `max_distance` für reine Distanz-Matches.
    """
    num_tracks = predicted_track_boxes_7d.shape[0]
    num_detections = current_detection_boxes_7d.shape[0]

    if num_tracks == 0 or num_detections == 0:
        return []

    cost_matrix = np.zeros((num_tracks, num_detections), dtype=np.float32)

    for i in range(num_tracks):
        for j in range(num_detections):
            track_box = predicted_track_boxes_7d[i]
            det_box = current_detection_boxes_7d[j]

            # 1. Distanzkosten (normalisiert)
            center_dist = np.linalg.norm(track_box[:2] - det_box[:2]) # XY-Distanz der Zentren
            # Normalisiere Distanzkosten auf [0,1]. Wenn center_dist > max_dist_norm, dann cost = 1.
            # Vermeide Division durch 0, falls max_center_dist_for_norm sehr klein ist.
            norm_dist_cost = min(center_dist / max_center_dist_for_norm, 1.0) if max_center_dist_for_norm > 1e-6 else 1.0
            
            # 2. IOU-Kosten (1 - IOU)
            iou = bev_iou(track_box, det_box)
            iou_cost = 1.0 - iou # Kosten: 0 für perfekte IOU, 1 für keine IOU

            # 3. Kombinierte Kosten
            combined_cost = W_DIST * norm_dist_cost + W_IOU * iou_cost
            cost_matrix[i, j] = combined_cost
            
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    
    matches = []
    for r, c in zip(row_idx, col_idx):
        cost = cost_matrix[r, c]
        if cost <= max_combined_cost: # Filter basierend auf den kombinierten Kosten
            matches.append((r, c, cost))
    return matches


# --- Platzhalter für die Track-Klasse und MultiObjectTracker ---
# (Diese wurden in Schritt 2.2.1 und werden in 2.2.3 angepasst)
# Hier nur, um das Skript syntaktisch korrekt zu halten für einen isolierten Test der Matching-Funktion.

_NEXT_TRACK_ID = 0 # Globale Variable für Track IDs

class Track:
    def __init__(self, initial_box_world: np.ndarray, initial_velocity_world: np.ndarray, 
                 track_id: int, initial_dt: float = 0.05, initial_score: float = 0.5):
        self.id = track_id
        self.box_world: np.ndarray = initial_box_world.copy()
        self.velocity_world: np.ndarray = initial_velocity_world.copy()
        self.age: int = 0
        self.hits: int = 1
        self.time_since_update: float = 0.0
        self.dt_history: List[float] = [initial_dt]
        self.last_match_cost: Optional[float] = None
        self.confidence_score: float = initial_score

    def predict(self, dt: float) -> np.ndarray: # Gibt jetzt (7,) Box zurück
        predicted_box_world = self.box_world.copy()
        current_center = predicted_box_world[:3]
        # Einfache Prädiktion des Zentrums
        if self.velocity_world is not None and self.velocity_world.shape == (3,):
             # Annahme: predict_future_positions ist global verfügbar oder hier definiert
             # Für den Test hier fügen wir eine Dummy-Implementierung ein,
             # da die eigentliche in der vollständigen tracking_utils.py steht.
            def predict_future_positions_dummy(pos, vel, delta_t):
                return pos + vel * delta_t
            predicted_center_flat = predict_future_positions_dummy(
                current_center, # .reshape(1,3) nicht nötig wenn predict_future_positions angepasst ist
                self.velocity_world, # .reshape(1,3) nicht nötig
                dt
            ) # .flatten() nicht nötig
            predicted_box_world[:3] = predicted_center_flat
        return predicted_box_world

    def update(self, matched_box_world: np.ndarray, dt_for_this_update: float, 
               match_cost: float, max_match_distance_for_score_norm: float, # Beibehaltung für Konsistenz
               new_velocity_world: Optional[np.ndarray] = None):
        
        previous_center_for_vel_calc = self.box_world[:3].copy()
        self.box_world = matched_box_world.copy()
        if new_velocity_world is not None and new_velocity_world.shape == (3,):
            self.velocity_world = new_velocity_world.copy()
        elif dt_for_this_update > 1e-6:
            self.velocity_world = (self.box_world[:3] - previous_center_for_vel_calc) / dt_for_this_update
        
        self.age = 0
        self.hits += 1
        self.time_since_update = 0
        self.dt_history.append(dt_for_this_update)
        if len(self.dt_history) > 5: self.dt_history.pop(0)
        self.last_match_cost = match_cost
        
        # Die Konfidenzberechnung muss ggf. angepasst werden, wenn die Kosten jetzt anders skaliert sind.
        # `max_match_distance_for_score_norm` war für Distanzkosten.
        # Für kombinierte Kosten [0,1] könnte man es direkter verwenden oder anders normalisieren.
        # Hier vorerst eine einfache Annahme:
        self.confidence_score = max(0.0, 1.0 - match_cost) # Da combined_cost schon ~[0,1] ist


class MultiObjectTracker:
    def __init__(self, max_age: int, min_hits_for_output: int, 
                 match_max_combined_cost: float, # Neuer Name für den Parameter
                 center_dist_norm_factor: float = 10.0):
        global _NEXT_TRACK_ID
        _NEXT_TRACK_ID = 0 
        self.tracks: List[Track] = []
        self.max_age = max_age
        self.min_hits_for_output = min_hits_for_output
        self.match_max_combined_cost = match_max_combined_cost # Schwellenwert für kombinierte Kosten
        self.center_dist_norm_factor = center_dist_norm_factor # Für Normalisierung der Distanz in Kostenfunktion
        print(f"MultiObjectTracker (mit IOU) initialisiert: max_age={max_age}, min_hits={min_hits_for_output}, max_combined_cost={match_max_combined_cost:.2f}")

    def _get_next_id(self) -> int:
        global _NEXT_TRACK_ID
        _NEXT_TRACK_ID += 1
        return _NEXT_TRACK_ID

    def update(self, current_detections_world_7d: np.ndarray, 
                     current_velocities_world: Optional[np.ndarray], 
                     dt: float):
        
        # 1. Prädiziere nächste Position für alle bestehenden Tracks
        predicted_track_boxes_list = []
        track_indices_for_prediction = [] # Behält den Index des Tracks in self.tracks
        for i, track_obj in enumerate(self.tracks):
            predicted_box_7d = track_obj.predict(dt) # Liefert jetzt volle 7D Box
            predicted_track_boxes_list.append(predicted_box_7d)
            track_indices_for_prediction.append(i)
            track_obj.time_since_update += dt # Erhöhe Zeit seit letztem Update für alle Tracks

        if not predicted_track_boxes_list:
            predicted_track_boxes_np = np.zeros((0,7), dtype=np.float32)
        else:
            predicted_track_boxes_np = np.array(predicted_track_boxes_list, dtype=np.float32)

        # 2. Assoziiere Detektionen mit prädizierten Tracks
        # Die Funktion erwartet jetzt volle 7D Boxen
        matches_with_cost = _match_detections_to_tracks_hungarian(
            predicted_track_boxes_np,
            current_detections_world_7d, # Übergebe volle 7D Detektionen
            max_combined_cost=self.match_max_combined_cost, # Verwende den neuen Schwellenwert
            max_center_dist_for_norm=self.center_dist_norm_factor
        )
        
        matched_track_original_indices = set()
        matched_detection_indices = set()

        # 3. Aktualisiere gematchte Tracks
        for pred_track_list_idx, det_idx, cost_of_match in matches_with_cost:
            original_track_idx = track_indices_for_prediction[pred_track_list_idx]
            track_to_update = self.tracks[original_track_idx]
            
            det_velocity = None
            if current_velocities_world is not None and current_velocities_world.shape[0] > det_idx:
                det_velocity = current_velocities_world[det_idx]
            
            # Die Track.update Methode verwendet max_match_distance_for_score_norm.
            # Da unsere Kosten jetzt anders sind (ca. 0-1), müssen wir überlegen, wie der Score berechnet wird.
            # Die Track.update Methode wurde angepasst, um `1.0 - match_cost` zu verwenden.
            track_to_update.update(
                current_detections_world_7d[det_idx], 
                dt, 
                match_cost=cost_of_match,
                max_match_distance_for_score_norm=1.0, # Da Kosten schon ~[0,1], ist 1.0 hier passend
                new_velocity_world=det_velocity
            )
            
            matched_track_original_indices.add(original_track_idx)
            matched_detection_indices.add(det_idx)

        # 4. Behandle ungematchte Tracks und Detektionen
        for i, track_obj in enumerate(self.tracks):
            if i not in matched_track_original_indices:
                track_obj.age += 1
                # Optional: Reduziere Konfidenz bei Nicht-Match
                # track_obj.confidence_score *= 0.9 

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
                    initial_score=0.5 # Standard-Startscore
                )
                self.tracks.append(new_track)

        # 5. Entferne alte Tracks und bereite Ausgabe vor
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


# --- Testfunktion für die neue Kostenmatrix-Logik ---
def test_cost_matrix_logic():
    print("Starte Test der Kostenmatrix-Logik...")

    # Szenario 1: Zwei Boxen, die sich gut überlappen und nah sind
    track1 = np.array([0, 0, 0, 4, 2, 1, 0.0], dtype=np.float32) # x,y,z, l,w,h, yaw
    det1   = np.array([0.5, 0.5, 0, 4, 2, 1, 0.0], dtype=np.float32)
    
    # Szenario 2: Zwei Boxen, weit entfernt, kein Überlapp
    track2 = np.array([10, 10, 0, 4, 2, 1, 0.0], dtype=np.float32)
    det2   = np.array([0, 0, 0, 4, 2, 1, 0.0], dtype=np.float32)

    # Szenario 3: Boxen mit unterschiedlichem Yaw, aber guter Überlappung
    track3 = np.array([0, 0, 0, 4, 2, 1, np.pi/4], dtype=np.float32) # 45 Grad Yaw
    det3   = np.array([0.2, 0.2, 0, 4, 2, 1, np.pi/4 + 0.1], dtype=np.float32) # Leichte Abweichung

    # Szenario 4: Boxen mit gleicher Position, aber eine viel kleiner (sollte hohe IOU haben)
    track4 = np.array([5, 5, 0, 4, 2, 1, 0.0], dtype=np.float32)
    det4   = np.array([5, 5, 0, 2, 1, 1, 0.0], dtype=np.float32) # Kleinere Box

    predicted_boxes = np.array([track1, track2, track3, track4])
    detected_boxes  = np.array([det1, det2, det3, det4])

    # Parameter für die Kostenfunktion
    max_cost_threshold = 0.7 # Beispielhafter Schwellenwert für kombinierte Kosten
    norm_dist_factor = 15.0  # Beispielhafter Normalisierungsfaktor für Distanz

    print(f"\nParameter: W_DIST={W_DIST}, W_IOU={W_IOU}, MaxCombinedCost={max_cost_threshold}, NormDistFactor={norm_dist_factor}")

    # Berechne Kostenmatrix manuell für einzelne Paare zum Debuggen
    print("\n--- Einzelne Kostenberechnungen ---")
    for i, p_box in enumerate(predicted_boxes):
        for j, d_box in enumerate(detected_boxes):
            center_d = np.linalg.norm(p_box[:2] - d_box[:2])
            norm_d_cost = min(center_d / norm_dist_factor, 1.0)
            iou_val = bev_iou(p_box, d_box)
            iou_c = 1.0 - iou_val
            comb_c = W_DIST * norm_d_cost + W_IOU * iou_c
            print(f"Track {i+1} vs Det {j+1}: CenterDist={center_d:.2f}, NormDistCost={norm_d_cost:.2f}, IOU={iou_val:.2f}, IOUCost={iou_c:.2f} => CombinedCost={comb_c:.3f}")


    print("\n--- Volle Kostenmatrix und Hungarian Matching ---")
    matches = _match_detections_to_tracks_hungarian(
        predicted_boxes,
        detected_boxes,
        max_combined_cost=max_cost_threshold,
        max_center_dist_for_norm=norm_dist_factor
    )

    print(f"\nGefundene Matches (track_idx, detection_idx, cost):")
    if matches:
        for match in matches:
            print(f"  Track Index {match[0]} <-> Detection Index {match[1]}, Kosten: {match[2]:.3f}")
    else:
        print("  Keine Matches gefunden unterhalb des Schwellenwerts.")

    # Test mit einem MultiObjectTracker (vereinfacht)
    print("\n--- Test mit MultiObjectTracker Instanz ---")
    tracker = MultiObjectTracker(max_age=3, min_hits_for_output=1, 
                                 match_max_combined_cost=max_cost_threshold,
                                 center_dist_norm_factor=norm_dist_factor)
    
    # Frame 1: Initialisiere Tracks mit den ersten zwei "predicted_boxes" als Detektionen
    # Annahme: Velocities sind Nullen für diesen Test
    initial_velocities1 = np.zeros((2,3), dtype=np.float32)
    print("\nTracker Update - Frame 1 (Initialisierung mit track1, track3 als Detektionen):")
    output_frame1 = tracker.update(np.array([track1, track3]), initial_velocities1, dt=0.1)
    print(f"Tracker Output Frame 1: {len(output_frame1)} Tracks")
    for t_out in output_frame1: print(f"  {t_out}")
    
    # Frame 2: Versuche, die Detektionen det1 und det3 zu matchen
    # Die Tracks im Tracker wurden durch predict() leicht verschoben (wenn velocity != 0 wäre)
    # Für diesen Test bleiben die prädizierten Positionen der Tracks gleich wie ihre Initialisierung,
    # da die Dummy-predict_future_positions mit Null-Geschwindigkeit arbeitet.
    current_detections_frame2 = np.array([det1, det3])
    velocities_frame2 = np.zeros((2,3), dtype=np.float32)
    print("\nTracker Update - Frame 2 (Matching mit det1, det3):")
    output_frame2 = tracker.update(current_detections_frame2, velocities_frame2, dt=0.1)
    print(f"Tracker Output Frame 2: {len(output_frame2)} Tracks")
    for t_out in output_frame2: print(f"  {t_out}")
    
    print("\nTest der Kostenmatrix-Logik abgeschlossen.")

if __name__ == '__main__':
    # Dieser Block wird ausgeführt, wenn das Skript direkt gestartet wird.
    # Nützlich für isolierte Tests der Matching-Funktion.
    test_cost_matrix_logic()
