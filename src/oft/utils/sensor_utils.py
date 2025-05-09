#!/usr/bin/env python3
"""
Sensor-Utilities: Farbwahl und Annotation-Filterung nach Sensortyp.
"""

from typing import List, Dict

def get_sensor_box_color(ann_record: Dict) -> tuple:
    """
    Wählt basierend auf 'num_lidar_pts' und 'num_radar_pts' eine BGR-Farbe:
      - Orange: beide Sensoren (BGR (0,165,255))
      - Blau:   nur Lidar    (BGR (255,0,0))
      - Pink:   nur Radar    (BGR (203,192,255))
      - Grau:   keiner       (BGR (128,128,128))
    """
    lidar = ann_record.get('num_lidar_pts', 0)
    radar = ann_record.get('num_radar_pts', 0)
    if lidar > 0 and radar > 0:
        return (0, 165, 255)    # Orange
    elif lidar > 0:
        return (255, 0, 0)      # Blau
    elif radar > 0:
        return (203, 192, 255)  # Pink
    else:
        return (128, 128, 128)  # Grau

def filter_annotations_by_sensor(
    ann_tokens: List[str],
    ts,
    sensor_type: str
) -> List[str]:
    """
    Filtert eine Liste von Annotation-Tokens nach Sensortyp:
      - 'lidar' → nur Annotationen mit num_lidar_pts > 0
      - 'radar' → nur Annotationen mit num_radar_pts > 0
      - 'both'  → nur Annotationen mit beidem > 0
      - 'any'   → jede Annotation, die von mindestens einem Sensor gesehen wurde
      - sonst  → alle Annotationen (kein Filter)
    """
    filtered = []
    for token in ann_tokens:
        ann = ts.get('sample_annotation', token)
        lidar = ann.get('num_lidar_pts', 0)
        radar = ann.get('num_radar_pts', 0)
        if sensor_type == 'lidar' and lidar > 0:
            filtered.append(token)
        elif sensor_type == 'radar' and radar > 0:
            filtered.append(token)
        elif sensor_type == 'both' and lidar > 0 and radar > 0:
            filtered.append(token)
        elif sensor_type == 'any' and (lidar > 0 or radar > 0):
            filtered.append(token)
        elif sensor_type not in ['lidar', 'radar', 'both', 'any']:
            filtered.append(token)
    return filtered