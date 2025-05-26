#!/usr/bin/env python3
import argparse
import numpy as np
from truckscenes import TruckScenes
from truckscenes.utils.geometry_utils import BoxVisibility # Import für BoxVisibility
from oft.utils.config import load_config

def print_stage1_gt_for_sample_idx(config_path: str, target_sample_idx: int):
    """
    Lädt GT-Boxen für einen target_sample_idx, wie es stage1_visualize_gt.py tun würde,
    und gibt ihre Details in Weltkoordinaten aus.
    """
    cfg = load_config(config_path)
    dcfg = cfg["dataset"]
    # vcfg = cfg["visualization"] # Nicht direkt für GT-Laden hier benötigt, aber für Kontext
    # rcfg = cfg["render"] # Für visibility, falls wir es exakt nachbilden wollen

    print(f"INFO: Initialisiere TruckScenes für Stage1 GT-Box-Analyse...")
    ts = TruckScenes(version=str(dcfg["version"]).strip(), dataroot=str(dcfg["dataroot"]))
    
    if not (0 <= target_sample_idx < len(ts.sample)):
        print(f"FEHLER: target_sample_idx ({target_sample_idx}) ist außerhalb des gültigen Bereichs (0-{len(ts.sample)-1}).")
        return
    
    sample_record = ts.sample[target_sample_idx]
    sample_token = sample_record["token"]
    
    print(f"\n--- Stage1-Style Ground Truth Boxen für Sample-Token: {sample_token} (Index: {target_sample_idx}) ---")
    
    # Hole die Annotation-Tokens für den aktuellen Sample, genau wie in B3 und debug_gt_boxes_for_sample.py
    # stage1_visualize_gt.py macht das implizit über ts.get_sample_data in render_sample_boxes.
    # Um die *Weltkoordinaten* zu bekommen, verwenden wir ts.get_box für jeden Annotation-Token des Samples.
    
    annotation_tokens = sample_record["anns"]
    if not annotation_tokens:
        print("  Keine Annotationen für diesen Sample-Token gefunden.")
        return

    print(f"  {len(annotation_tokens)} Annotation-Tokens gefunden für diesen Sample.")
    # Optional: Filterung nach Sichtbarkeit, wie es render_sample_boxes tun könnte.
    # box_vis_config = cfg.get("render", {}).get("box_visibility", "ANY")
    # try:
    #     visibility_filter = BoxVisibility[box_vis_config.upper()]
    # except KeyError:
    #     visibility_filter = BoxVisibility.ANY
    # print(f"  (Hinweis: Sichtbarkeitsfilter aus Config wäre: {visibility_filter})")
    # Die tatsächliche Filterung nach Sichtbarkeit in ts.get_sample_data() ist komplexer,
    # da sie sich auf die Projektion in Kamerabilder bezieht.
    # Für eine reine Ausgabe der Welt-GT-Boxen dieses Frames ist das Filtern hier nicht trivial.
    # ts.get_box() selbst filtert nicht nach Sichtbarkeit.

    gt_boxes_world = []
    for ann_token in annotation_tokens:
        box = ts.get_box(ann_token) # Liefert DevBox in Weltkoordinaten
        gt_boxes_world.append(box)

    if not gt_boxes_world:
        print("  Keine GT-Boxen nach dem Laden für diesen Sample-Token.")
        return

    print(f"\n  Ausgabe der {len(gt_boxes_world)} GT-Boxen in Weltkoordinaten (wie von ts.get_box()):")
    for i, box in enumerate(gt_boxes_world):
        ann_record = ts.get("sample_annotation", box.token) # box.token ist der ann_token
        instance_record = ts.get("instance", ann_record["instance_token"])
        category_record = ts.get("category", instance_record["category_token"])

        print(f"\n  GT Box #{i+1} (aus ts.get_box):")
        print(f"    Annotation Token: {box.token}")
        print(f"    Kategorie:        {category_record['name']}")
        print(f"    Position (Welt):  {np.array2string(box.center, precision=3, separator=', ')}")
        print(f"    Größe (w,l,h):    {np.array2string(box.wlh, precision=3, separator=', ')}")
        print(f"    Yaw (Welt, Rad):  {box.orientation.yaw_pitch_roll[0]:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zeigt GT-Box-Details für einen Sample-Index (Stage1-Logik-inspiriert).")
    parser.add_argument("-c", "--pipeline", required=False, default="config/pipeline.yaml",
                        help="Pfad zur pipeline.yaml.")
    parser.add_argument("-i", "--sample_idx", required=True, type=int,
                        help="Der Index des Samples, dessen GT-Boxen angezeigt werden sollen.")
    
    script_args = parser.parse_args()
    print_stage1_gt_for_sample_idx(config_path=script_args.pipeline, target_sample_idx=script_args.sample_idx)
    