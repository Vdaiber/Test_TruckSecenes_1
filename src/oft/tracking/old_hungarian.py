# Datei: hungarian.py

import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian_match(prev_positions, curr_positions, prev_velocities=None,
                    ego_transform=None, dt=0.0, max_distance=10.0):
    """
    Führt Matching zwischen zwei Sets von Objektpositionen durch (z.B. von vorherigem zu aktuellem Frame).
    
    Argumente:
        prev_positions (np.ndarray): Array der Form (N,2) mit [x,y] der Objekte im früheren Frame.
        curr_positions (np.ndarray): Array der Form (M,2) mit [x,y] der Objekte im aktuellen Frame.
        prev_velocities (np.ndarray oder None): Optional (N,2) Geschwindigkeitsvektoren [vx, vy] im vorherigen Frame.
        ego_transform (np.ndarray oder None): Optional 3x3 oder 4x4 Transformationsmatrix (Ego-Bewegung) von prev nach curr.
        dt (float): Zeitdifferenz zwischen den Frames (Sekunden) für die Forward-Projektion.
        max_distance (float): Maximale Distanz zur Berücksichtigung eines Matches.
    
    Rückgabe:
        matches (List[(int, int)]): Liste der Zuordnungen (index_prev, index_curr).
    """
    N = prev_positions.shape[0]
    M = curr_positions.shape[0]
    if N == 0 or M == 0:
        return []

    # 1) Ego-Bewegungstransformation auf vorherige Positionen anwenden (optional)
    if ego_transform is not None:
        # Homogene Koordinaten [x, y, 1]
        prev_hom = np.hstack([prev_positions, np.ones((N,1))])
        transformed = (ego_transform @ prev_hom.T).T
        prev_positions = transformed[:, :2]

    # 2) Forward-Projektion mit Objektgeschwindigkeiten (optional)
    if prev_velocities is not None:
        # Zugabe von v*dt auf die Positionen
        prev_positions = prev_positions + prev_velocities * dt

    # 3) Kostenmatrix basierend auf euklidischer Distanz
    cost_matrix = np.linalg.norm(
        prev_positions[:, np.newaxis, :] - curr_positions[np.newaxis, :, :],
        axis=2
    )  # Form (N, M)

    # Große Kosten für Distanzen über dem Schwellenwert
    cost_matrix[cost_matrix > max_distance] = 1e6

    # 4) Hungarian Matching (Minimieren der Kosten)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < 1e5:  # akzeptiert nur, wenn nicht als ungültig markiert
            matches.append((i, j))
    return matches

if __name__ == "__main__":
    # Beispielnutzung / Test
    prev_pos = np.array([[0,0],[5,0],[10,0]])
    curr_pos = np.array([[1,0],[5.5,0],[9.8,0]])
    prev_vel = np.array([[1,0],[0.5,0],[0,-0.2]])
    # Ohne Ego-Transform: Positionsdifferenz plus Velocity*dt
    matches = hungarian_match(prev_pos, curr_pos, prev_velocities=prev_vel, dt=1.0, max_distance=2.0)
    print("Matches (prev->curr):", matches)