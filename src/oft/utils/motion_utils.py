# Datei: motion_utils.py

import json
import numpy as np
# (Optional: scipy.spatial.transform für genauere Rotation, hier vereinfachend nur in Z-Achse)
try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    R = None

class MotionUtils:
    def __init__(self, ego_pose_file, ego_motion_chassis_file=None, ego_motion_cabin_file=None):
        """
        Lädt Ego-Pose-Daten und (optional) Ego-Motion-Daten.
        Erwartete Struktur: JSON-Tabellen mit Zeitstempeln und Pose/Drehung.
        """
        # Lade Ego-Posen (und konvertiere in Struktur {timestamp: [x,y,z, qx,qy,qz,qw]})
        with open(ego_pose_file, 'r') as f:
            data = json.load(f)
        # Annahme: data ist entweder Liste oder Dict mit key 'ego_pose'
        entries = data.get('ego_pose', data) if isinstance(data, dict) else data
        self.poses = {}
        for entry in entries:
            ts = entry.get('timestamp', entry.get('time', None))
            if ts is None:
                continue
            # Extrahiere Position und Orientierung
            tx, ty, tz = entry.get('translation', entry.get('position', [None]*3))
            qx, qy, qz, qw = entry.get('rotation', [None]*4)
            # Speichere in Dictionary
            self.poses[ts] = {
                'translation': np.array([tx, ty, tz]),
                'rotation': np.array([qx, qy, qz, qw])  # quaternion [x,y,z,w]
            }
        # (Optional) Ego-Motion-Daten können geladen und integriert werden, wenn nötig
        # (z.B. als Listen von (timestamp, dx,dy,dz, dyaw) für Chassis/Cabin).
        self.chassis_motion = None
        self.cabin_motion = None
        # Hier nicht weiterverwendet, kann bei Bedarf implementiert werden.

    def get_relative_transform(self, t_from, t_to):
        """
        Berechnet die relative Transformationsmatrix von Zeit t_from nach t_to.
        Nur in XY-Ebene, z wird ignoriert, Rotation nur um Z-Achse (Yaw).
        Rückgabe: 3x3 homogene Transformationsmatrix.
        """
        pose0 = self.poses.get(t_from)
        pose1 = self.poses.get(t_to)
        if pose0 is None or pose1 is None:
            raise KeyError(f"Ego-Pose für {t_from} oder {t_to} nicht gefunden.")
        # Translationen
        p0 = pose0['translation']
        p1 = pose1['translation']
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        # Yaw-Winkel aus Quaternions berechnen (Annahme: nur Rotation um Z)
        if R is not None:
            r0 = R.from_quat([pose0['rotation'][0], pose0['rotation'][1],
                              pose0['rotation'][2], pose0['rotation'][3]])
            r1 = R.from_quat([pose1['rotation'][0], pose1['rotation'][1],
                              pose1['rotation'][2], pose1['rotation'][3]])
            yaw0 = r0.as_euler('xyz')[2]
            yaw1 = r1.as_euler('xyz')[2]
        else:
            # Falls scipy nicht verfügbar, grobe Schätzung per Atan2 (nur falls Quaternion vorhanden)
            q0 = pose0['rotation']; q1 = pose1['rotation']
            # Achtung: Annahme: Quaternion im Format [x,y,z,w]
            yaw0 = 2 * np.arctan2(q0[2], q0[3])
            yaw1 = 2 * np.arctan2(q1[2], q1[3])
        dyaw = yaw1 - yaw0
        # Homogene Transformationsmatrix (Rotation + Translation)
        cos_y = np.cos(dyaw)
        sin_y = np.sin(dyaw)
        transform = np.array([
            [cos_y, -sin_y, dx],
            [sin_y,  cos_y, dy],
            [   0.0,   0.0, 1.0]
        ])
        return transform

    @staticmethod
    def apply_transform(positions, transform):
        """
        Wendet eine 3x3 homogene Transformationsmatrix auf Punkte (x,y) an.
        positions: np.ndarray der Form (N,2).
        """
        N = positions.shape[0]
        hom = np.hstack([positions, np.ones((N,1))])  # (N,3)
        trans = (transform @ hom.T).T
        return trans[:, :2]

    @staticmethod
    def forward_project(positions, velocities, dt):
        """
        Extrapoliert Positionen anhand konstanter Geschwindigkeit für Zeitschritt dt.
        positions und velocities sind Arrays der Form (N,2).
        """
        return positions + velocities * dt

if __name__ == "__main__":
    # Beispielnutzung / Test
    # Angenommen, ego_pose.json enthält:
    # [{'timestamp': 0.0, 'translation':[0,0,0], 'rotation':[0,0,0,1]},
    #  {'timestamp': 1.0, 'translation':[1,0,0], 'rotation':[0,0,0,1]}]
    # Dann ist die Transform zwischen t=0 und t=1:
    import io
    ego_pose_json = io.StringIO(json.dumps({
        "ego_pose": [
            {"timestamp": 0.0, "translation": [0, 0, 0], "rotation": [0, 0, 0, 1]},
            {"timestamp": 1.0, "translation": [1, 0, 0], "rotation": [0, 0, 0, 1]}
        ]
    }))
    with open('ego_pose.json', 'w') as f: f.write(ego_pose_json.getvalue())
    mu = MotionUtils('ego_pose.json')
    T = mu.get_relative_transform(0.0, 1.0)
    print("Transform 0->1:", T)
    # Anwenden auf Punkt (1,1)
    pt = np.array([[1.0, 1.0]])
    pt_new = mu.apply_transform(pt, T)
    print("Punkt (1,1) in neuem Koordinatensystem:", pt_new)