# src/oft/tracking/hungarian.py
import yaml
import numpy as np
from scipy.optimize import linear_sum_assignment

class HungarianTracker:
    def __init__(self):
        config = yaml.safe_load(open('pipeline.yaml'))
        self.max_age = config['tracking']['max_age']
        self.iou_threshold = config['tracking']['iou_threshold']
        self.tracks = []  # Liste aktiver Tracks

    def predict(self, dt):
        # Bewegungsprojektion: Für jeden Track Position anhand letzter Geschwindigkeit schätzen.
        for track in self.tracks:
            track.position += track.velocity * dt  # einfacher CV-Model
            track.age += dt

    def update(self, detections, timestamp):
        """Aktualisiert Tracker mit neuen Detektionen (Ground-Truth)."""
        # 1) Vorhersage für Tracks (Zeitdifferenz seit letztem Update)
        self.predict(dt=timestamp - self.last_timestamp)
        self.last_timestamp = timestamp

        # 2) Kostenmatrix über IoU berechnen
        cost = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou = compute_bev_iou(track.box, det.box)
                cost[i, j] = 1 - iou  # Minimierungsproblem

        # 3) Matching per Hungarian
        row_ind, col_ind = linear_sum_assignment(cost)
        matched, unmatched_dets = set(), set(range(len(detections)))
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < (1 - self.iou_threshold):
                # Weisen Track i Detektion j zu
                self.tracks[i].update_with(detections[j])
                matched.add(i); unmatched_dets.discard(j)

        # 4) Neue Tracks für unverknüpfte Detektionen
        for j in unmatched_dets:
            new_track = Track(init_detection=detections[j])
            self.tracks.append(new_track)

        # 5) Entfernen alter Tracks nach max_age
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]