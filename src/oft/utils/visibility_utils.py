import numpy as np
from typing import Tuple
from truckscenes import TruckScenes # Import TruckScenes class
from truckscenes.utils.data_classes import Box as DevBox
from truckscenes.utils.geometry_utils import box_in_image, BoxVisibility
from pyquaternion import Quaternion

def transform_world_box_to_sensor_coords(world_box: DevBox, 
                                         ego_pose_record: dict, 
                                         calibrated_sensor_record: dict) -> DevBox:
    """
    Transformiert eine DevBox von Weltkoordinaten in Sensorkoordinaten.
    Die Logik ist an truckscenes.TruckScenes.boxes_to_sensor angelehnt,
    arbeitet aber auf einer Kopie, um die Originalbox nicht zu verändern.
    """
    # Kopiere die Box, um das Original nicht zu verändern
    sensor_box = world_box.copy() # Erstellt eine neue DevBox Instanz mit denselben Werten

    # Von Welt zu Ego
    sensor_box.translate(-np.array(ego_pose_record['translation']))
    sensor_box.rotate(Quaternion(ego_pose_record['rotation']).inverse)
    
    # Von Ego zu Sensor
    sensor_box.translate(-np.array(calibrated_sensor_record['translation']))
    sensor_box.rotate(Quaternion(calibrated_sensor_record['rotation']).inverse)
    
    return sensor_box

def is_world_box_visible_in_camera_view(world_box: DevBox,
                                        ts_instance: TruckScenes, 
                                        camera_sd_token: str,
                                        image_width: int,
                                        image_height: int,
                                        box_vis_level: BoxVisibility = BoxVisibility.ANY
                                        ) -> bool:
    """
    Prüft, ob eine 3D-Box (gegeben in Weltkoordinaten) für eine gegebene Kamera 
    potenziell sichtbar ist, indem sie in Kamerakoordinaten transformiert und 
    dann truckscenes.utils.geometry_utils.box_in_image verwendet wird.
    """
    sd_record = ts_instance.get('sample_data', camera_sd_token)
    cs_record = ts_instance.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    # Ego-Pose zum Zeitpunkt des Kamera-Frames holen
    ego_pose_record = ts_instance.get('ego_pose', sd_record['ego_pose_token']) 
    
    camera_intrinsic = np.array(cs_record['camera_intrinsic'])
    imsize = (image_width, image_height) # (width, height) für box_in_image

    # Transformiere die Welt-Box in die Koordinaten des aktuellen Kamerasensors
    box_in_sensor_frame = transform_world_box_to_sensor_coords(world_box, ego_pose_record, cs_record)

    # Verwende die Devkit-Funktion box_in_image
    # box_in_image prüft intern auch, ob die Box-Ecken vor der Kamera liegen (z > 0.1m)
    return box_in_image(box_in_sensor_frame, camera_intrinsic, imsize, vis_level=box_vis_level)