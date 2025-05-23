#!/usr/bin/env python3
import numpy as np
from scipy.optimize import linear_sum_assignment
# Die folgenden Imports werden nicht mehr direkt benötigt, wenn wir alles im Welt-KS machen
# und die Eingaben von stage3 bereits Weltkoordinaten sind.
# from oft.utils.motion_utils import (
#     box_centers_sensor_to_world,
#     predict_world_centers, # Diesen behalten wir eventuell oder nutzen eine einfachere Version
#     box_centers_world_to_sensor,
# )

# Einfache Prädiktionsfunktion, falls `predict_world_centers` aus motion_utils zu komplex ist
# oder spezifische Annahmen trifft, die hier nicht passen.
# Deine motion_utils.predict_world_centers ist aber simpel: return world_centers + velocities * dt
# Das ist genau das, was wir brauchen.
from oft.utils.motion_utils import predict_world_centers


def temporal_hungarian_match(
    prev_centers_xy,     # np.ndarray[N_prev,2] # von stage3: prev_xyz_world[:, :2] -> WIRD JETZT IGNORIERT
    curr_centers_xy,     # np.ndarray[N_curr,2] # von stage3: curr_xyz_world[:, :2] -> DIES SIND WELT-XY für Kosten
    prev_centers_3d,     # np.ndarray[N_prev,3] # von stage3: prev_xyz_world -> WELT-XYZ (t-1)
    prev_velocities,     # np.ndarray[N_prev,3] or None # von stage3: prev_vel_world -> WELT-VEL (t-1)
    ego_pose_prev,       # np.ndarray[4,4] # von stage3: T_world_from_ego_prev -> Nicht mehr direkt hier benötigt
    ego_pose_curr,       # np.ndarray[4,4] # von stage3: T_world_from_ego_curr -> Nicht mehr direkt hier benötigt
    sensor_extrinsic,    # np.ndarray[4,4] # von stage3: E_ego_from_sensor -> Nicht mehr direkt hier benötigt
    dt,                  # float
    max_distance         # float
):
    """
    Führt Hungarian Matching zwischen vorhergesagten Welt-Positionen aus Frame t-1
    und aktuellen Welt-Positionen in Frame t durch.
    Alle Berechnungen finden im Weltkoordinatensystem statt.
    """
    print(f"    INSIDE temporal_hungarian_match:")
    print(f"      Input prev_centers_3d (world @ t-1) ({prev_centers_3d.shape}): {prev_centers_3d[:2].tolist() if prev_centers_3d.shape[0]>0 else 'leer'}")
    print(f"      Input curr_centers_xy (world @ t) ({curr_centers_xy.shape}): {curr_centers_xy[:2].tolist() if curr_centers_xy.shape[0]>0 else 'leer'}")
    print(f"      Input prev_velocities (world) ({prev_velocities.shape if prev_velocities is not None else 'None'}): {prev_velocities[:2].tolist() if prev_velocities is not None and prev_velocities.shape[0]>0 else 'N/A oder leer'}")
    print(f"      Input dt: {dt:.4f}s")

    # Schritt 1: Prädiziere die Welt-Positionen von t-1 nach t
    if prev_velocities is not None and prev_centers_3d.shape[0] > 0:
        if prev_velocities.shape[0] != prev_centers_3d.shape[0]:
            print(f"      WARNUNG: Shape-Mismatch zwischen prev_centers_3d ({prev_centers_3d.shape[0]}) und prev_velocities ({prev_velocities.shape[0]}). Setze Velocities auf Null.")
            velocities_for_prediction = np.zeros_like(prev_centers_3d)
        else:
            velocities_for_prediction = prev_velocities
        
        predicted_world_centers_at_t = predict_world_centers(prev_centers_3d, velocities_for_prediction, dt)
    elif prev_centers_3d.shape[0] > 0 :
        print("      INFO: Keine Geschwindigkeiten vorhanden, Prädiktion ist gleich prev_centers_3d.")
        predicted_world_centers_at_t = np.copy(prev_centers_3d)
    else:
        print("      INFO: Keine prev_centers_3d zum Prädizieren.")
        predicted_world_centers_at_t = np.zeros((0,3), dtype=prev_centers_3d.dtype)

    print(f"      Predicted world_centers_at_t ({predicted_world_centers_at_t.shape}): {predicted_world_centers_at_t[:2].tolist() if predicted_world_centers_at_t.shape[0]>0 else 'leer'}")
    
    if predicted_world_centers_at_t.shape[0] == 0 or curr_centers_xy.shape[0] == 0:
        print("      INFO: Keine prädizierten oder aktuellen Boxen für Kostenmatrix. Gebe 0 Matches zurück.")
        return []

    pred_xy_world = predicted_world_centers_at_t[:, :2]
    
    cost_matrix = np.linalg.norm(
        pred_xy_world[:, None, :] - curr_centers_xy[None, :, :],
        axis=-1
    )
    
    # Optional: Gib einen kleinen Teil der Kostenmatrix aus, um die Werte zu sehen
    # print(f"      Cost Matrix (Ausschnitt, shape {cost_matrix.shape}):\n{cost_matrix[:min(5, cost_matrix.shape[0]),:min(5, cost_matrix.shape[1])]}")


    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    matches = []
    print(f"      Gefundene Matches (vor Distanzfilter):") # DEBUG
    for r, c in zip(row_idx, col_idx):
        cost = cost_matrix[r, c]
        # DEBUG: Gib die Details jedes potenziellen Matches aus
        # print(f"        Potenzieller Match: prev_idx={r}, curr_idx={c}, Distanz/Kosten={cost:.4f}m") 
        if cost <= max_distance:
            matches.append((r, c))
            print(f"        => Akzeptierter Match: prev_idx={r} (pred_xy: {pred_xy_world[r].tolist()}) mit curr_idx={c} (curr_xy: {curr_centers_xy[c].tolist()}), Kosten={cost:.4f}m") # NEUE DEBUG-AUSGABE
        # else:
            # print(f"        => Abgelehnter Match: prev_idx={r}, curr_idx={c}, Kosten={cost:.4f}m > max_dist={max_distance:.2f}m")
            
    print(f"      Final matches (nach Distanzfilter, max_dist={max_distance:.2f}m): {matches}")
    return matches