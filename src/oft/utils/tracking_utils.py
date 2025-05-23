#!/usr/bin/env python3
import numpy as np
from scipy.optimize import linear_sum_assignment

def predict_future_positions(
    prev_positions: np.ndarray,
    velocities: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Lineare Projektion im Welt-KS: p' = p + v·dt
    - prev_positions: (N,3) XYZ
    - velocities:     (N,3)
    - dt:             Zeitdifferenz in Sekunden
    → (N,3)
    """
    return prev_positions + velocities * dt

def hungarian_match_world(
    prev_boxes: np.ndarray,
    curr_boxes: np.ndarray,
    velocities: np.ndarray = None,
    dt: float = 1.0,
    max_distance: float = 1.0
) -> list[tuple[int,int]]:
    """
    Hungarian-Matching zwischen:


    - prev_boxes: (N_prev,7) [x,y,z,w,l,h,yaw]
    - curr_boxes: (N_curr,7)
    - velocities: (N_prev,3) optional
    - dt:         float
    - max_distance: float (Meter)

    Rückgabe: Liste von (i,j)-Paare, i aus prev, j aus curr.
    """
    # Zentren extrahieren
    prev_centers = prev_boxes[:, :3]
    curr_centers = curr_boxes[:, :3]

    # 1) Prognose p' = p + v·dt
    if velocities is not None:
        pred = predict_future_positions(prev_centers, velocities, dt)
    else:
        pred = prev_centers

    # 2) Kostenmatrix (nur XY-Abstand)
    cost = np.linalg.norm(
        pred[:, None, :2] - curr_centers[None, :, :2],
        axis=-1
    )  # Form: (N_prev, N_curr)

    # 3) Assignment
    row_idx, col_idx = linear_sum_assignment(cost)

    # 4) Threshold-Filter
    matches = [
        (i, j)
        for i, j in zip(row_idx, col_idx)
        if cost[i, j] <= max_distance
    ]
    return matches