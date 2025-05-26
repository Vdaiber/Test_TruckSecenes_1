#!/usr/bin/env python3
import argparse
import numpy as np
from truckscenes import TruckScenes
from truckscenes.utils.geometry_utils import BoxVisibility 
from oft.utils.config import load_config

def print_stage1_comparable_gt_boxes(config_path: str, target_sample_idx: int):
    """
    Lädt GT-Boxen für einen target_sample_idx, versucht die Filterung von 
    stage1_visualize_gt.py (via ts.get_sample_data) nachzuahmen,
    und gibt die Details der entsprechenden Weltkoordinaten-Boxen aus.
    """
    cfg = load_config(config_path)
    dcfg = cfg["dataset"]
    vcfg = cfg["visualization"]
    rcfg = cfg["render"]

    camera_channel = vcfg.get("camera_channel", "CAMERA_FRONT")
    box_vis_config = rcfg.get("box_visibility", "ANY")
    try:
        visibility_filter = BoxVisibility[box_vis_config.upper()]
    except KeyError:
        print(f"WARNUNG: Ungültiger box_visibility Wert '{box_vis_config}'. Verwende 'ANY'.")
        visibility_filter = BoxVisibility.ANY

    print(f"INFO: Initialisiere TruckScenes für Stage1-vergleichbare GT-Box-Analyse...")
    ts = TruckScenes(version=str(dcfg["version"]).strip(), dataroot=str(dcfg["dataroot"]))
    
    if not (0 <= target_sample_idx < len(ts.sample)):
        print(f"FEHLER: target_sample_idx ({target_sample_idx}) ist außerhalb des gültigen Bereichs (0-{len(ts.sample)-1}).")
        return
    
    sample_record = ts.sample[target_sample_idx]
    sample_token = sample_record["token"]
    
    print(f"\n--- Stage1-Vergleichbare Ground Truth Boxen für Sample-Token: {sample_token} (Index: {target_sample_idx}) ---")
    print(f"--- Verwendeter Kamera-Kanal: {camera_channel}, Sichtbarkeitsfilter: {visibility_filter.name} ---")

    # Hole sample_data_token für den spezifizierten Kamerakanal
    if camera_channel not in sample_record["data"]:
        print(f"FEHLER: Kamera-Kanal '{camera_channel}' nicht im Sample {sample_token} gefunden.")
        return
    sd_token = sample_record["data"][camera_channel]

    # Verwende ts.get_sample_data(), da dies die Filterung nach Sichtbarkeit vornimmt
    # und Boxen in Kamerakoordinaten liefert. Die zurückgegebenen Box-Objekte
    # enthalten aber den originalen 'token' der sample_annotation.
    try:
        _, boxes_cam_filtered, _ = ts.get_sample_data(sd_token, box_vis_level=visibility_filter)
    except Exception as e:
        print(f"FEHLER beim Aufruf von ts.get_sample_data für sd_token {sd_token}: {e}")
        return

    if not boxes_cam_filtered:
        print("  Keine Boxen nach Sichtbarkeitsfilterung durch ts.get_sample_data() für diesen Kamerakanal gefunden.")
        return

    print(f"\n  {len(boxes_cam_filtered)} Boxen nach Kamera-Sichtbarkeitsfilterung ({visibility_filter.name}) gefunden.")
    print(f"  Details der entsprechenden Weltkoordinaten-Boxen:")

    for i, cam_box in enumerate(boxes_cam_filtered):
        ann_token = cam_box.token # Dies ist der sample_annotation_token
        
        # Hole die vollständigen Weltkoordinaten-Details für diese Annotation
        world_box = ts.get_box(ann_token)
        ann_record = ts.get("sample_annotation", ann_token)
        instance_record = ts.get("instance", ann_record["instance_token"])
        category_record = ts.get("category", instance_record["category_token"])

        print(f"\n  GT Box #{i+1} (entspricht gefilterter Kamera-Box):")
        print(f"    Annotation Token: {ann_token}")
        print(f"    Kategorie:        {category_record['name']}")
        print(f"    Position (Welt):  {np.array2string(world_box.center, precision=3, separator=', ')}")
        print(f"    Größe (w,l,h):    {np.array2string(world_box.wlh, precision=3, separator=', ')}")
        print(f"    Yaw (Welt, Rad):  {world_box.orientation.yaw_pitch_roll[0]:.3f}")
        print(f"    Sichtbarkeit (laut Annotation): {ann_record.get('visibility_token', 'N/A')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zeigt GT-Box-Details (Weltkoordinaten), die der Filterung von stage1_visualize_gt entsprechen.")
    parser.add_argument("-c", "--pipeline", required=False, default="config/pipeline.yaml",
                        help="Pfad zur pipeline.yaml.")
    parser.add_argument("-i", "--sample_idx", required=True, type=int,
                        help="Der Index des Samples, dessen GT-Boxen angezeigt werden sollen.")
    
    script_args = parser.parse_args()
    print_stage1_comparable_gt_boxes(config_path=script_args.pipeline, target_sample_idx=script_args.sample_idx)
    