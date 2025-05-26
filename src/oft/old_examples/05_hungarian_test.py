#!/usr/bin/env python3
import numpy as np
from oft.tracking.old_hungarian import hungarian_match

def main():
    # zwei projizierte Punkte und drei aktuelle
    prev = np.array([[0,0],[5,0]])
    curr = np.array([[0.5,0],[5.2,0],[10,0]])
    matches = hungarian_match(prev, curr, prev_velocities=None, ego_transform=None, dt=1.0, max_distance=2.0)
    print("Matches (idx_prev→idx_curr):", matches)
    # Erwartung: [(0,0),(1,1)] – der dritte Curr bleibt unmatched
if __name__=="__main__":
    main()