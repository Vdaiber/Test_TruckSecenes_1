#!/usr/bin/env python3
import argparse
import numpy as np
from truckscenes import TruckScenes
from oft.utils.config import load_config # Annahme: load_config ist verfügbar

def print_gt_boxes_for_sample(config_path: str, target_sample_idx: int):
    """
    Lädt einen bestimmten Sample aus TruckScenes und gibt die Details
    seiner Ground-Truth-Annotationen auf der Konsole aus.
    """
    cfg = load_config(config_path)
    dcfg = cfg["dataset"]

    print(f"INFO: Initialisiere TruckScenes für GT-Box-Analyse...")
    ts = TruckScenes(version=str(dcfg["version"]).strip(), dataroot=str(dcfg["dataroot"]))
    
    if not (0 <= target_sample_idx < len(ts.sample)):
        print(f"FEHLER: target_sample_idx ({target_sample_idx}) ist außerhalb des gültigen Bereichs (0-{len(ts.sample)-1}).")
        return
    
    sample_record = ts.sample[target_sample_idx]
    sample_token = sample_record["token"]
    
    print(f"\n--- Ground Truth Boxen für Sample-Token: {sample_token} (Index: {target_sample_idx}) ---")
    
    if not sample_record["anns"]:
        print("  Keine Annotationen für diesen Sample-Token gefunden.")
        return

    for i, ann_token in enumerate(sample_record["anns"]):
        ann_record = ts.get("sample_annotation", ann_token)
        instance_record = ts.get("instance", ann_record["instance_token"])
        category_record = ts.get("category", instance_record["category_token"])
        
        # Hole die Box-Daten direkt mit ts.get_box() für Weltkoordinaten
        box = ts.get_box(ann_token) # truckscenes.utils.data_classes.Box

        print(f"\n  GT Box #{i+1}:")
        print(f"    Annotation Token: {ann_token}")
        print(f"    Instance Token:   {ann_record['instance_token']}")
        print(f"    Kategorie:        {category_record['name']} (Token: {instance_record['category_token']})")
        print(f"    Position (Welt, center x,y,z): {np.array2string(box.center, precision=3, separator=', ')}")
        print(f"    Größe (Welt, w,l,h):          {np.array2string(box.wlh, precision=3, separator=', ')}")
        print(f"    Orientierung (Welt, Quat w,x,y,z): {np.array2string(box.orientation.elements, precision=3, separator=', ')}")
        print(f"    Yaw (Welt, Radiant):           {box.orientation.yaw_pitch_roll[0]:.3f}")
        # print(f"    Velocity (aus DevBox, falls vorhanden): {box.velocity}") # DevBox hat standardmäßig keine Velocity, muss separat berechnet/geholt werden
        print(f"    Anzahl LiDAR-Punkte: {ann_record['num_lidar_pts']}")
        print(f"    Anzahl Radar-Punkte: {ann_record['num_radar_pts']}")
        visibility_token = ann_record.get('visibility_token')
        if visibility_token: # Stelle sicher, dass der Token nicht leer ist
            visibility_record = ts.get('visibility', visibility_token)
            print(f"    Sichtbarkeit:       {visibility_record.get('level')} ({visibility_record.get('description')})")
        else:
            print(f"    Sichtbarkeit:       Nicht spezifiziert")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zeigt Ground Truth Box-Details für einen Sample-Index.")
    parser.add_argument("-c", "--pipeline", required=False, default="config/pipeline.yaml",
                        help="Pfad zur pipeline.yaml.")
    parser.add_argument("-i", "--sample_idx", required=True, type=int,
                        help="Der Index des Samples, dessen GT-Boxen angezeigt werden sollen.")
    
    script_args = parser.parse_args()
    print_gt_boxes_for_sample(config_path=script_args.pipeline, target_sample_idx=script_args.sample_idx)