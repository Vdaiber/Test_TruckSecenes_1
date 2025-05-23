#!/usr/bin/env python3
import numpy as np
from scipy.optimize import linear_sum_assignment
from oft.utils.motion_utils import (
    box_centers_sensor_to_world,
    predict_world_centers,
    box_centers_world_to_sensor,
)

def temporal_hungarian_match(
    prev_centers_xy,     # np.ndarray[N_prev,2]
    curr_centers_xy,     # np.ndarray[N_curr,2]
    prev_centers_3d,     # np.ndarray[N_prev,3]
    prev_velocities,     # np.ndarray[N_prev,3] or None
    ego_pose_prev,       # np.ndarray[4,4]
    ego_pose_curr,       # np.ndarray[4,4]
    sensor_extrinsic,    # np.ndarray[4,4]
    dt,                  # float
    max_distance         # float
):
    """
    Führt Hungarian Matching zwischen vorhergesagten Positionen aus Frame t-1
    und aktuellen Positionen in Frame t durch.
    """
    # 1) Sensor→Ego→Welt (t-1)
    world_prev = box_centers_sensor_to_world(prev_centers_3d, sensor_extrinsic, ego_pose_prev)

    # 2) Optional: Welt‐Punkte mit Velocity verschieben
    if prev_velocities is not None:
        world_prev = predict_world_centers(world_prev, prev_velocities, dt)

    # 3) Welt→Ego→Sensor (t)
    sensor_pred = box_centers_world_to_sensor(world_prev, sensor_extrinsic, ego_pose_curr)

    # 4) Kostenmatrix: euklidische Distanz der 2D‐Zentren
    pred_xy = sensor_pred[:, :2]  # (N_prev,2)
    cost = np.linalg.norm(
        pred_xy[:, None, :] - curr_centers_xy[None, :, :],
        axis=-1
    )  # Form (N_prev, N_curr)

    # 5) Assignment
    row_idx, col_idx = linear_sum_assignment(cost)

    # 6) Filter nach Threshold
    matches = [
        (i, j) for (i, j) in zip(row_idx, col_idx)
        if cost[i, j] <= max_distance
    ]
    return matches