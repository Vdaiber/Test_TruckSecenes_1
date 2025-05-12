# debug_truckscenes.py
#!/usr/bin/env python3
import os
import pprint

from truckscenes import TruckScenes

def debug_scene(dataroot, version, scene_idx=0):
    ts = TruckScenes(version=version, dataroot=dataroot)
    print(f"Loaded TruckScenes v{version!r} @ {dataroot!r}")
    print(f"→ Anzahl Szenen: {len(ts.scene)}\n")

    # 1) Szene aufschlüsseln
    scene = ts.scene[scene_idx]
    pprint.pprint({"scene_record": scene})
    first_sample = scene["first_sample_token"]
    last_sample  = scene["last_sample_token"]
    print(f"\nFirst sample token: {first_sample!r}")
    print(f"Last  sample token: {last_sample!r}\n")
    
    # 2) Ein einzelnes Sample
    sample = ts.get("sample", first_sample)
    pprint.pprint({"sample_record": sample})
    
    # 3) sample["data"] Einträge
    print("\n→ sample['data'] Kanäle:")
    for chan, sd_token in sample["data"].items():
        sd = ts.get("sample_data", sd_token)
        print(f" {chan:20s} → token={sd_token}, modality={sd['sensor_modality']}, file={sd['filename']}")
        
        # calibrated_sensor & ego_pose
        calib   = ts.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ego     = ts.get("ego_pose",         sd["ego_pose_token"])
        print(f"    calibrated_sensor_token: {sd['calibrated_sensor_token']} → {list(calib.keys())}")
        print(f"    ego_pose_token:           {sd['ego_pose_token']} → {list(ego.keys())}")
    
    # 4) sample["anns"]
    print("\n→ sample['anns'] (Annotations):", sample["anns"])
    for ann in sample["anns"]:
        ann_rec = ts.get("sample_annotation", ann)
        print(f"  • {ann}: category={ann_rec['category_name']}, #lidar={ann_rec['num_lidar_pts']}, #radar={ann_rec['num_radar_pts']}")
    
    # 5) Wetter? (Gibt es nicht intern, aber hier werft ihr mal einen Blick in alle Keys)
    print("\n--- Alle Keys in scene/sample/sample_data/etc. ---")
    print("Scene keys:       ", list(scene.keys()))
    print("Sample keys:      ", list(sample.keys()))
    print("Sample_data keys: ", list(ts.get("sample_data", sample["data"][list(sample["data"])[0]]).keys()))
    # calibrated_sensor & ego_pose keys
    print("Calibrated keys:  ", list(calib.keys()))
    print("Ego_pose   keys:  ", list(ego.keys()))

if __name__ == "__main__":
    # Passt hier dataroot & version an eure Config an
    dataroot = "/app/datasets"
    version  = "v1.0-mini"
    debug_scene(dataroot, version, scene_idx=0)
    