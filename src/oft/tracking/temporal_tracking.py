#!/usr/bin/env python3
"""
Temporal Hungarian Matching:

Diese Funktion führt ein Hungarian-Matching zwischen 
vorhergesagten (historischen) und aktuellen Box-Zentren durch.
Alle Parameter – insbesondere dt und max_distance – müssen
vom Aufrufer (z.B. scripts/stage3_track_history.py) 
aus der Pipeline-Konfiguration übergeben werden.

Args:
    prev_centers_xy     (np.ndarray[N_prev,2]):  x,y der Box-Zentren im Sensor-Koordsystem bei t-1
    curr_centers_xy     (np.ndarray[N_curr,2]):  x,y der Box-Zentren im Sensor-Koordsystem bei t
    prev_centers_3d     (np.ndarray[N_prev,3]):  x,y,z der Box-Zentren im Sensor-Koordsystem bei t-1
    prev_velocities     (np.ndarray[N_prev,3] or None):
                                                vx,vy,vz in Welt-Koordsystem (optional)
    ego_pose_prev       (np.ndarray[4,4]):      Ego-Pose (Welt←Ego) bei t-1
    ego_pose_curr       (np.ndarray[4,4]):      Ego-Pose (Welt←Ego) bei t
    sensor_extrinsic    (np.ndarray[4,4]):      Sensor-Extrinsik (Ego←Sensor)
    dt                  (float):                Zeitdifferenz zwischen den Frames in Sekunden
    max_distance        (float):                Maximale Distanz für ein Match (in Sensor-Koordsystem)

Returns:
    List[Tuple[int,int]]: Liste der (i,j)-Matches, i aus prev, j aus curr,
                          nur für Paare mit Distanz ≤ max_distance.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from oft.utils.motion_utils import (
    box_centers_sensor_to_world,
    predict_world_centers,
    box_centers_world_to_sensor,
)

def temporal_hungarian_match(
    prev_centers_xy,
    curr_centers_xy,
    prev_centers_3d,
    prev_velocities,
    ego_pose_prev,
    ego_pose_curr,
    sensor_extrinsic,
    dt,
    max_distance
):
    # 1) Transform Sensor→Ego→Welt (t-1)
    world_prev = box_centers_sensor_to_world(
        prev_centers_3d,
        sensor_extrinsic,
        ego_pose_prev
    )

    # 2) Optional: Welt-Punkte linear mit Velocity verschieben
    if prev_velocities is not None:
        world_prev = predict_world_centers(world_prev, prev_velocities, dt)

    # 3) Transform Welt→Ego→Sensor (t)
    sensor_pred = box_centers_world_to_sensor(
        world_prev,
        sensor_extrinsic,
        ego_pose_curr
    )

    # 4) Kostenmatrix: euklidische Distanz zwischen prognostizierten xy und aktuellen xy
    pred_xy = sensor_pred[:, :2]   # (N_prev,2)
    cost = np.linalg.norm(
        pred_xy[:, None, :] - curr_centers_xy[None, :, :],
        axis=-1
    )  # Form (N_prev, N_curr)

    # 5) Hungarian-Algorithmus
    row_idx, col_idx = linear_sum_assignment(cost)

    # 6) Nur Paare mit Distanz ≤ max_distance behalten
    matches = [
        (i, j) for (i, j) in zip(row_idx, col_idx)
        if cost[i, j] <= max_distance
    ]
    return matches