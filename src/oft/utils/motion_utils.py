import json
import numpy as np
from typing import Dict, Any, Optional, List
try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    R = None

class MotionUtils:
    def __init__(self, ego_pose_file: str, ego_motion_chassis_file: Optional[str]=None, ego_motion_cabin_file: Optional[str]=None):
        """
        Lädt Ego-Pose-Daten aus JSON und speichert sie intern.
        """
        with open(ego_pose_file, 'r') as f:
            data = json.load(f)
        entries = data.get('ego_pose', data) if isinstance(data, dict) else data
        self.poses: Dict[Any, Dict[str, np.ndarray]] = {}
        for entry in entries:
            ts = entry.get('timestamp', entry.get('time', None))
            if ts is None:
                continue
            tx, ty, tz = entry.get('translation', entry.get('position', [None]*3))
            qx, qy, qz, qw = entry.get('rotation', [None]*4)
            self.poses[ts] = {
                'translation': np.array([tx, ty, tz], dtype=np.float32),
                'rotation':    np.array([qx, qy, qz, qw], dtype=np.float32)
            }

    def get_relative_transform(self, t_from: float, t_to: float) -> np.ndarray:
        """
        Berechnet 3×3 Homogene Transformationsmatrix von t_from nach t_to in XY-Ebene.
        """
        pose0 = self.poses.get(t_from)
        pose1 = self.poses.get(t_to)
        if pose0 is None or pose1 is None:
            raise KeyError(f"Ego-Pose für {t_from} oder {t_to} nicht gefunden.")
        p0 = pose0['translation']; p1 = pose1['translation']
        dx, dy = p1[0]-p0[0], p1[1]-p0[1]

        if R is not None:
            r0 = R.from_quat(pose0['rotation'])
            r1 = R.from_quat(pose1['rotation'])
            yaw0 = r0.as_euler('xyz')[2]; yaw1 = r1.as_euler('xyz')[2]
        else:
            q0, q1 = pose0['rotation'], pose1['rotation']
            yaw0 = 2*np.arctan2(q0[2], q0[3]); yaw1 = 2*np.arctan2(q1[2], q1[3])
        dyaw = yaw1 - yaw0
        cos_y, sin_y = np.cos(dyaw), np.sin(dyaw)

        # 3×3
        return np.array([
            [cos_y, -sin_y, dx],
            [sin_y,  cos_y, dy],
            [  0.0,    0.0, 1.0]
        ], dtype=np.float32)

    @staticmethod
    def apply_transform(positions: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """
        Wendet 3×3 Transform auf (N,2)-Punkte an.
        """
        N = positions.shape[0]
        hom = np.hstack([positions, np.ones((N,1),dtype=np.float32)])
        trans = (transform @ hom.T).T
        return trans[:, :2]

    @staticmethod
    def forward_project(positions: np.ndarray, velocities: np.ndarray, dt: float) -> np.ndarray:
        """
        Lineare Projektion: p' = p + v·dt
        """
        return positions + velocities * dt

def box_centers_sensor_to_world(centers: np.ndarray, E: np.ndarray, T_ego: np.ndarray) -> np.ndarray:
    """
    Konvertiert (N,3) Sensor-Koords → Welt-Koords.
    """
    N = centers.shape[0]
    c_h = np.hstack([centers, np.ones((N,1),dtype=np.float32)])   # (N,4)
    ego_pts = (E @ c_h.T)                                         # (4,N)
    world_pts = (T_ego @ ego_pts)                                 # (4,N)
    return world_pts[:3].T                                        # (N,3)

def predict_world_centers(world_centers: np.ndarray, velocities: np.ndarray, dt: float) -> np.ndarray:
    """
    p' = p + v·dt im Welt-KS.
    """
    return world_centers + velocities * dt

def box_centers_world_to_sensor(world_centers: np.ndarray, E: np.ndarray, T_ego: np.ndarray) -> np.ndarray:
    """
    Konvertiert (N,3) Welt-Koords → Sensor-Koords.
    """
    N = world_centers.shape[0]
    w_h = np.hstack([world_centers, np.ones((N,1),dtype=np.float32)])
    ego_inv = np.linalg.inv(T_ego)
    ego_pts = (ego_inv @ w_h.T)
    E_inv = np.linalg.inv(E)
    sensor_pts = (E_inv @ ego_pts)
    return sensor_pts[:3].T