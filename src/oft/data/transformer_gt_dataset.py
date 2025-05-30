# src/oft/data/transformer_gt_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional

from truckscenes import TruckScenes
from truckscenes.utils.splits import create_splits_scenes # Zum Laden der Szenen-Splits
from oft.utils.common_utils import parse_scene_description # Für scene_meta

class ObjectFusionGTDataset(Dataset):
    """
    Dataset-Klasse zum Laden von Ground-Truth (GT) 3D-Objektdaten aus TruckScenes.
    Bereitet Daten für ein Transformer-Modell vor, wobei GT-Objekte als
    "perfekte Detektionen" eines konzeptionellen Einzelsensors behandelt werden.
    Lädt umfassende Metadaten und eine Historie von GT-Boxen für zukünftige Erweiterungen.
    """
    def __init__(self,
                 dataroot: str,
                 version: str,
                 split_name: str, 
                 pipeline_config: Dict[str, Any],
                 verbose: bool = True):
        """
        Args:
            dataroot (str): Pfad zum Root-Verzeichnis des TruckScenes-Datensatzes.
            version (str): Version des TruckScenes-Datensatzes (z.B. 'v1.0-mini').
            split_name (str): Name des zu ladenden Datensplits (z.B. 'train', 'val', 'mini_train').
            pipeline_config (Dict[str, Any]): Die geladene pipeline.yaml Konfiguration.
            verbose (bool): Ob Statusmeldungen ausgegeben werden sollen.
        """
        super().__init__()
        self.dataroot = dataroot
        self.version = version
        self.split_name = split_name
        self.cfg = pipeline_config 
        self.verbose = verbose

        # Lese history_window aus der Konfiguration oder verwende einen Standardwert
        dataset_cfg = self.cfg.get('dataset', {})
        self.history_window = int(dataset_cfg.get('history_window_transformer', 3))

        if self.verbose:
            print(f"Initializing ObjectFusionGTDataset for split '{self.split_name}' with history_window={self.history_window}...")

        self.ts = TruckScenes(version=self.version, dataroot=self.dataroot, verbose=self.verbose)

        scene_name_splits = create_splits_scenes()
        if self.split_name not in scene_name_splits:
            raise ValueError(f"Split '{self.split_name}' not found in TruckScenes splits. Available: {list(scene_name_splits.keys())}")
        
        self.scene_names_in_split = scene_name_splits[self.split_name]
        if self.verbose:
            print(f"Found {len(self.scene_names_in_split)} scenes for split '{self.split_name}'.")

        self.sample_tokens: List[str] = []
        for scene_record in self.ts.scene:
            if scene_record['name'] in self.scene_names_in_split:
                current_sample_token = scene_record['first_sample_token']
                while current_sample_token:
                    self.sample_tokens.append(current_sample_token)
                    sample = self.ts.get('sample', current_sample_token)
                    if not sample: break
                    current_sample_token = sample['next']
        
        if not self.sample_tokens:
            print(f"WARNING: No sample tokens found for split '{self.split_name}'. Please check dataset and split configuration.")
        elif self.verbose:
            print(f"Collected {len(self.sample_tokens)} sample tokens for split '{self.split_name}'.")

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_token = self.sample_tokens[idx]
        sample_record = self.ts.get('sample', sample_token)
        
        # --- Metadaten für den aktuellen Frame ---
        scene_record = self.ts.get('scene', sample_record['scene_token'])
        scene_meta = parse_scene_description(scene_record['description'])

        sample_data_records: Dict[str, Dict] = {}
        calibrated_sensor_records: Dict[str, Dict] = {}
        for channel, sd_token in sample_record['data'].items():
            sd_rec = self.ts.get('sample_data', sd_token)
            sample_data_records[channel] = sd_rec
            if sd_rec:
                cs_rec = self.ts.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
                calibrated_sensor_records[channel] = cs_rec
        
        ego_pose_record = self.ts.getclosest('ego_pose', sample_record['timestamp'])
        ego_translation_world = np.array(ego_pose_record['translation'], dtype=np.float32)
        ego_rotation_world_quat = np.array(ego_pose_record['rotation'], dtype=np.float32)

        # --- GT-Detektionen für den aktuellen Frame ---
        gt_detections_list: List[Dict[str, Any]] = []
        current_boxes_list: List[np.ndarray] = []
        current_velocities_list: List[np.ndarray] = []
        current_class_labels_list: List[int] = []
        current_instance_tokens_list: List[str] = []
        current_annotation_tokens_list: List[str] = []

        annotation_records_dict: Dict[str, Dict] = {}
        instance_records_dict: Dict[str, Dict] = {}
        attribute_records_dict: Dict[str, List[Dict]] = {}
        visibility_records_dict: Dict[str, Dict] = {}

        annotation_tokens = sample_record.get('anns', [])
        for ann_token in annotation_tokens:
            try:
                ann_record = self.ts.get('sample_annotation', ann_token)
                annotation_records_dict[ann_token] = ann_record
                
                instance_token = ann_record['instance_token']
                instance_record = self.ts.get('instance', instance_token)
                instance_records_dict[instance_token] = instance_record
                
                category_record = self.ts.get('category', instance_record['category_token'])
                class_label = int(category_record['index'])
                
                attribute_tokens = ann_record.get('attribute_tokens', [])
                attribute_records_dict[ann_token] = [self.ts.get('attribute', at) for at in attribute_tokens if self.ts.get('attribute', at)]
                
                visibility_token = ann_record.get('visibility_token')
                if visibility_token:
                    visibility_records_dict[ann_token] = self.ts.get('visibility', visibility_token)

                box_obj = self.ts.get_box(ann_token)
                if box_obj is None: continue
                
                box_world = np.array([
                    box_obj.center[0], box_obj.center[1], box_obj.center[2],
                    box_obj.wlh[0], box_obj.wlh[1], box_obj.wlh[2],
                    box_obj.orientation.yaw_pitch_roll[0]
                ], dtype=np.float32)

                velocity_world_raw = self.ts.box_velocity(ann_token)
                velocity_world = np.nan_to_num(velocity_world_raw, nan=0.0).astype(np.float32)

                gt_detections_list.append({
                    "box_world": box_world, "velocity_world": velocity_world,
                    "class_label": class_label, "instance_token": instance_token,
                    "annotation_token": ann_token
                })
                current_boxes_list.append(box_world)
                current_velocities_list.append(velocity_world)
                current_class_labels_list.append(class_label)
                current_instance_tokens_list.append(instance_token)
                current_annotation_tokens_list.append(ann_token)

            except Exception as e:
                if self.verbose: print(f"Error processing annotation {ann_token} in sample {sample_token}: {e}")
                continue
        
        current_boxes_7d = np.array(current_boxes_list, dtype=np.float32) if current_boxes_list else np.zeros((0, 7), dtype=np.float32)
        current_velocities_3d = np.array(current_velocities_list, dtype=np.float32) if current_velocities_list else np.zeros((0, 3), dtype=np.float32)
        current_class_labels = np.array(current_class_labels_list, dtype=np.int64) if current_class_labels_list else np.zeros((0,), dtype=np.int64)

        # --- History Frames laden ---
        history_boxes_7d: List[Optional[np.ndarray]] = []
        history_sample_tokens: List[Optional[str]] = []
        
        prev_hist_token = sample_record['prev']
        for _ in range(self.history_window):
            if not prev_hist_token:
                history_boxes_7d.append(None)
                history_sample_tokens.append(None)
                continue

            hist_sample_record = self.ts.get('sample', prev_hist_token)
            if not hist_sample_record:
                history_boxes_7d.append(None)
                history_sample_tokens.append(None)
                break 
            
            history_sample_tokens.append(prev_hist_token)
            hist_ann_tokens = hist_sample_record.get('anns', [])
            single_hist_frame_boxes: List[np.ndarray] = []
            for hist_ann_token in hist_ann_tokens:
                try:
                    hist_box_obj = self.ts.get_box(hist_ann_token)
                    if hist_box_obj:
                        single_hist_frame_boxes.append(np.array([
                            hist_box_obj.center[0], hist_box_obj.center[1], hist_box_obj.center[2],
                            hist_box_obj.wlh[0], hist_box_obj.wlh[1], hist_box_obj.wlh[2],
                            hist_box_obj.orientation.yaw_pitch_roll[0]
                        ], dtype=np.float32))
                except Exception:
                    pass # Ignore errors in history frames for robustness
            
            if single_hist_frame_boxes:
                history_boxes_7d.append(np.array(single_hist_frame_boxes))
            else:
                history_boxes_7d.append(np.zeros((0,7), dtype=np.float32)) # Empty array if no boxes
            
            prev_hist_token = hist_sample_record['prev']

        history_boxes_7d.reverse() # Chronological order [t-H, ..., t-1]
        history_sample_tokens.reverse()

        history_padding_mask = np.array([
            1 if (h_boxes is not None and h_boxes.shape[0] > 0) else 0
            for h_boxes in history_boxes_7d
        ], dtype=bool)


        output = {
            "sample_token": sample_token,
            "timestamp": sample_record['timestamp'],
            "prev_sample_token": sample_record['prev'],
            "next_sample_token": sample_record['next'],
            "scene_token": sample_record['scene_token'],
            
            "ego_translation_world": ego_translation_world,
            "ego_rotation_world_quat": ego_rotation_world_quat,
            
            "sample_data_records": sample_data_records,
            "calibrated_sensor_records": calibrated_sensor_records,
            
            "annotation_records_dict": annotation_records_dict,
            "instance_records_dict": instance_records_dict,
            "attribute_records_dict": attribute_records_dict,
            "visibility_records_dict": visibility_records_dict,
            
            "scene_meta": scene_meta,
            
            "gt_detections": gt_detections_list,
            
            "current_boxes_7d": current_boxes_7d,
            "current_velocities_3d": current_velocities_3d,
            "current_class_labels": current_class_labels,
            "current_instance_tokens": current_instance_tokens_list,
            "current_annotation_tokens": current_annotation_tokens_list,

            "history_boxes_7d": history_boxes_7d,
            "history_sample_tokens": history_sample_tokens,
            "history_padding_mask": history_padding_mask,
        }
        
        return output

if __name__ == '__main__':
    print("Running ObjectFusionGTDataset example...")
    
    dummy_pipeline_config = {
        "dataset": {
            "dataroot": "/pfad/zu/deinem/truckscenes", 
            "version": "v1.0-mini",
            "history_window_transformer": 2 # Beispiel für History Window
        }
    }

    dataroot_path = dummy_pipeline_config["dataset"]["dataroot"]
    dataset_version = dummy_pipeline_config["dataset"]["version"]
    
    if dataroot_path == "/pfad/zu/deinem/truckscenes":
        print("BITTE ANPASSEN: 'dataroot_path' im Beispielcode auf deinen lokalen TruckScenes-Pfad setzen.")
    else:
        try:
            dataset = ObjectFusionGTDataset(
                dataroot=dataroot_path,
                version=dataset_version,
                split_name='mini_train', 
                pipeline_config=dummy_pipeline_config,
                verbose=True
            )

            if len(dataset) > 0:
                print(f"\nDataset initialized successfully. Number of samples in 'mini_train': {len(dataset)}")
                
                sample_data = dataset[0]
                print(f"\n--- Sample 0 ({sample_data['sample_token']}) ---")
                for key, value in sample_data.items():
                    if key == "gt_detections":
                        print(f"  {key}: {len(value)} detections")
                        if value: print(f"    First gt_detection: {value[0]}")
                    elif key == "history_boxes_7d":
                        print(f"  {key}: {len(value)} history frames")
                        for i, hist_frame in enumerate(value):
                            print(f"    Frame t-{len(value)-i}: {'No boxes' if hist_frame is None or hist_frame.shape[0] == 0 else str(hist_frame.shape)}")
                    elif isinstance(value, np.ndarray):
                        print(f"  {key}: shape {value.shape}")
                    elif isinstance(value, list):
                        print(f"  {key}: {len(value)} items")
                    elif isinstance(value, dict):
                         print(f"  {key}: {len(value.keys())} keys")
                    else:
                        print(f"  {key}: {value}")
            else:
                print("Dataset could be initialized, but no samples were found for 'mini_train'.")

        except FileNotFoundError as e:
            print(f"\nError during dataset initialization: {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()