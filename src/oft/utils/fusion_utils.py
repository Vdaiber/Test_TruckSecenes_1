# src/oft/utils/fusion_utils.py
import numpy as np
from typing import List, Dict, Any, Optional
import math

def calculate_track_distance(track1: Dict[str, Any], track2: Dict[str, Any], 
                             use_velocity_term: bool = False, 
                             velocity_weight: float = 0.5) -> float:
    """
    Berechnet eine Distanz/Kosten zwischen zwei Tracks.
    Aktuell: Euklidische Distanz der 3D-Box-Zentren.
    Optional: Gewichtete Hinzunahme der Geschwindigkeitsdifferenz.
    """
    center1 = np.array(track1.get("box_world", [0,0,0,0,0,0,0])[:3])
    center2 = np.array(track2.get("box_world", [0,0,0,0,0,0,0])[:3])
    
    # Sicherstellen, dass beide Center gültig sind (falls box_world fehlt oder zu kurz ist)
    if center1.shape[0] != 3 or center2.shape[0] != 3:
        return float('inf') # Hohe Kosten, falls Box-Daten unvollständig

    pos_distance = np.linalg.norm(center1 - center2)

    if use_velocity_term:
        # Sicherstellen, dass velocity_world existiert und korrekte Dimension hat
        vel1_data = track1.get("velocity_world", [0,0,0])
        vel2_data = track2.get("velocity_world", [0,0,0])
        if not (isinstance(vel1_data, (list, np.ndarray)) and len(vel1_data) >= 2 and 
                isinstance(vel2_data, (list, np.ndarray)) and len(vel2_data) >= 2):
            # Wenn Geschwindigkeitsdaten fehlen oder ungültig sind, nur Positionsdistanz verwenden
            return pos_distance

        vel1 = np.array(vel1_data[:2]) # XY Geschw.
        vel2 = np.array(vel2_data[:2]) # XY Geschw.
        vel_distance = np.linalg.norm(vel1 - vel2)
        
        # Kombinierte Distanz (einfache Gewichtung, kann verfeinert werden)
        return (1 - velocity_weight) * pos_distance + velocity_weight * vel_distance
    
    return pos_distance

def average_yaws_weighted(yaws: List[float], weights: List[float]) -> float:
    """
    Berechnet den gewichteten Mittelwert von Yaw-Winkeln (in Radiant).
    Behandelt den Sprung bei -pi/pi durch Mittelung von Einheitsvektoren.
    """
    if not yaws or not weights or len(yaws) != len(weights):
        # print("Warning: Yaws or weights list empty or mismatched lengths in average_yaws_weighted.")
        return np.mean(yaws) if yaws else 0.0 # Fallback oder Fehler werfen

    sum_sin = 0.0
    sum_cos = 0.0
    
    # Filtere Gewichte, die nicht positiv sind
    positive_weights_present = any(w > 1e-6 for w in weights)
    if not positive_weights_present: # Alle Gewichte sind effektiv null oder negativ
        return np.mean(yaws) if yaws else 0.0

    current_sum_weights = sum(w for w in weights if w > 1e-6) # Nur positive Gewichte für die Summe verwenden
    if current_sum_weights == 0: # Sollte durch positive_weights_present Check oben abgedeckt sein
        return np.mean(yaws) if yaws else 0.0

    for yaw, weight in zip(yaws, weights):
        if weight > 1e-6: # Berücksichtige nur Tracks mit signifikantem Gewicht
            # Normalisiere das Gewicht für die Vektorkomponenten-Mittelung
            normalized_weight = weight / current_sum_weights
            sum_sin += normalized_weight * np.sin(yaw) 
            sum_cos += normalized_weight * np.cos(yaw) 
            
    # Wenn sum_sin und sum_cos beide sehr klein sind (z.B. durch sich aufhebende Winkel mit gleichen Gewichten),
    # kann atan2 instabil werden oder einen willkürlichen Winkel zurückgeben.
    # Ein Fallback könnte der Yaw des Tracks mit dem höchsten Gewicht sein.
    if abs(sum_sin) < 1e-9 and abs(sum_cos) < 1e-9: # Engere Toleranz für Vektorlänge nahe Null
        # Fallback: Nehme den Yaw des Tracks mit dem höchsten positiven Gewicht
        if yaws and any(w > 1e-6 for w in weights): 
             # Finde den Index des höchsten positiven Gewichts
             max_positive_weight = -1.0
             idx_of_max_positive_weight = -1
             for idx, w_val in enumerate(weights):
                 if w_val > 1e-6 and w_val > max_positive_weight:
                     max_positive_weight = w_val
                     idx_of_max_positive_weight = idx
             
             if idx_of_max_positive_weight != -1:
                 return yaws[idx_of_max_positive_weight]
        return np.mean(yaws) if yaws else 0.0 # Generischer Fallback

    return np.arctan2(sum_sin, sum_cos)

def fuse_track_group(track_group: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Fusioniert eine Gruppe von Tracks zu einem einzigen Track-Dictionary.
    Verwendet gewichtete Mittelwerte basierend auf 'confidence_score'.
    """
    if not track_group:
        return None

    # Wähle den Referenz-Track basierend auf dem höchsten Konfidenzscore.
    # Wenn mehrere Tracks den gleichen höchsten Score haben, wird der erste davon genommen (durch `max`).
    ref_track = max(track_group, key=lambda t: t.get("confidence_score", 0.0))

    # Summe der Konfidenzscores für die Gewichtung. Nur positive Scores verwenden.
    confidences = [t.get("confidence_score", 0.0) for t in track_group]
    # Filtert Scores, die zu klein sind, um numerische Probleme zu vermeiden.
    valid_confidences_for_weighting = [c for c in confidences if c > 1e-6] 

    if not valid_confidences_for_weighting: # Wenn alle Scores effektiv null sind
        # Fallback: Gib den Referenz-Track (den mit dem höchsten Ursprungsscore, auch wenn er 0 ist) zurück,
        # aber aktualisiere seine Fusions-Metadaten.
        fused_fallback_dict = ref_track.copy() # Kopiere, um Original nicht zu ändern
        fused_fallback_dict["source_sensor"] = "fused_fallback"
        fused_fallback_dict["contributing_sensors"] = sorted(list(set(t.get("source_sensor", "unknown") for t in track_group)))
        fused_fallback_dict["num_fused_tracks"] = len(track_group)
        # Behalte den ursprünglichen (höchsten) Score, wenn alle anderen zu niedrig sind.
        fused_fallback_dict["confidence_score"] = ref_track.get("confidence_score", 0.0) 
        return fused_fallback_dict

    total_confidence_sum_for_weighting = sum(valid_confidences_for_weighting)

    fused_box_world_params = np.zeros(6, dtype=np.float32) # x,y,z,w,l,h
    fused_velocity_world = np.zeros(3, dtype=np.float32) # vx,vy,vz
    
    yaws_to_average = []
    yaw_weights = [] # Hier die originalen Scores > 1e-6 für average_yaws_weighted verwenden

    # Box-Parameter (außer Yaw) und Geschwindigkeiten mitteln
    for track in track_group:
        score = track.get("confidence_score", 0.0)
        if score <= 1e-6: # Überspringe Tracks mit vernachlässigbarem Score für die Mittelung
            continue

        weight_for_avg = score / total_confidence_sum_for_weighting
        
        # Translation (x,y,z) und Size (w,l,h)
        current_box_params = np.array(track.get("box_world", [0.0]*7)[:6])
        fused_box_world_params += weight_for_avg * current_box_params
        
        # Geschwindigkeit (vx,vy,vz)
        current_velocity = np.array(track.get("velocity_world", [0.0,0.0,0.0]))
        fused_velocity_world += weight_for_avg * current_velocity
        
        # Yaws und Scores (als Gewichte) für spätere Mittelung sammeln
        yaws_to_average.append(track.get("box_world", [0.0]*7)[6])
        yaw_weights.append(score) 

    fused_yaw = average_yaws_weighted(yaws_to_average, yaw_weights)
    
    final_fused_box_world = fused_box_world_params.tolist() + [fused_yaw]

    # Neuer Konfidenzscore: Hier nehmen wir das Maximum der beitragenden Scores.
    new_confidence_score = max(c for c in confidences) # Max aller Original-Confidences
    # Optional: Kleiner Bonus, wenn mehrere Tracks fusioniert wurden (aber nicht über 1.0)
    if len(valid_confidences_for_weighting) > 1: # Überprüfe anhand der für die Gewichtung genutzten Tracks
        new_confidence_score = min(1.0, new_confidence_score + 0.05 * (len(valid_confidences_for_weighting) -1))

    fused_track_dict = {
        "track_id": ref_track.get("track_id", -1), 
        "box_world": final_fused_box_world,
        "velocity_world": fused_velocity_world.tolist(),
        "age": ref_track.get("age", 0), 
        "hits": max(t.get("hits", 1) for t in track_group), 
        "confidence_score": round(new_confidence_score, 4),
        "last_match_cost": ref_track.get("last_match_cost"), 
        "source_sensor": "fused", 
        "contributing_sensors": sorted(list(set(t.get("source_sensor", "unknown") for t in track_group))), 
        "num_fused_tracks": len(track_group) 
    }
    return fused_track_dict