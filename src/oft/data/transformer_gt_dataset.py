# src/oft/data/transformer_gt_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict # NEU: defaultdict importiert

from truckscenes import TruckScenes
from truckscenes.utils.splits import create_splits_scenes # Zum Laden der Szenen-Splits
from oft.utils.common_utils import parse_scene_description # Für scene_meta
from oft.utils.config import load_config 
import os 

class ObjectFusionGTDataset(Dataset):
    """
    Dataset-Klasse zum Laden von Ground-Truth (GT) 3D-Objektdaten aus TruckScenes.
    Bereitet Daten für ein Transformer-Modell vor, wobei GT-Objekte als
    "perfekte Detektionen" eines konzeptionellen Einzelsensors behandelt werden.
    Lädt umfassende Metadaten und eine Historie von GT-Boxen.
    Erstellt Features für aktuelle und historische GT-Objekte als Input für den Encoder.
    """
    def __init__(self,
                 dataroot: str,
                 version: str,
                 split_name: str, 
                 pipeline_config: Dict[str, Any], 
                 verbose: bool = True):
        super().__init__()
        self.dataroot = dataroot
        self.version = version
        self.split_name = split_name
        self.pipeline_cfg = pipeline_config 
        self.verbose = verbose

        dataset_specific_cfg = self.pipeline_cfg.get('dataset', {})
        self.history_window = int(dataset_specific_cfg.get('history_window_transformer', 0)) 

        model_specific_cfg = self.pipeline_cfg.get('model', {})
        self.num_encoder_input_features = int(model_specific_cfg.get('num_input_features', 10)) 

        if self.verbose:
            print(f"Initializing ObjectFusionGTDataset for split '{self.split_name}' with history_window={self.history_window}, num_encoder_input_features={self.num_encoder_input_features}...")

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
                    sample_record_check = self.ts.get('sample', current_sample_token) 
                    if not sample_record_check: break 
                    current_sample_token = sample_record_check['next']
        
        if not self.sample_tokens:
            print(f"WARNING: No sample tokens found for split '{self.split_name}'. Please check dataset and split configuration.")
        elif self.verbose:
            print(f"Collected {len(self.sample_tokens)} sample tokens for split '{self.split_name}'.")

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def _get_full_box_data_for_token(self, sample_token: str) -> Tuple[List[Dict[str, Any]], Dict, Dict, Dict, Dict]:
        """
        Hilfsfunktion, um Box-Daten und zugehörige Metadaten für einen gegebenen Sample-Token zu extrahieren.
        Gibt gt_detections_list und die Dictionaries für annotations, instances, attributes, visibility zurück.
        """
        gt_detections_list: List[Dict[str, Any]] = []
        annotation_records_dict: Dict[str, Dict] = {}
        instance_records_dict: Dict[str, Dict] = {}
        attribute_records_dict: Dict[str, List[Dict]] = defaultdict(list) # Defaultdict für Attribute
        visibility_records_dict: Dict[str, Dict] = {}

        sample_record = self.ts.get('sample', sample_token)
        if not sample_record:
            return gt_detections_list, annotation_records_dict, instance_records_dict, attribute_records_dict, visibility_records_dict
            
        annotation_tokens = sample_record.get('anns', [])
        for ann_token in annotation_tokens:
            try:
                ann_record = self.ts.get('sample_annotation', ann_token)
                if not ann_record: continue
                annotation_records_dict[ann_token] = ann_record
                
                instance_token = ann_record['instance_token']
                instance_record = self.ts.get('instance', instance_token)
                if not instance_record: continue
                instance_records_dict[instance_token] = instance_record
                
                category_record = self.ts.get('category', instance_record['category_token'])
                if not category_record: continue
                class_label_idx = int(category_record['index']) 
                
                attribute_tokens = ann_record.get('attribute_tokens', [])
                for at in attribute_tokens:
                    attr_rec = self.ts.get('attribute', at)
                    if attr_rec:
                        attribute_records_dict[ann_token].append(attr_rec)
                
                visibility_token = ann_record.get('visibility_token')
                if visibility_token:
                    vis_rec = self.ts.get('visibility', visibility_token)
                    if vis_rec:
                        visibility_records_dict[ann_token] = vis_rec

                box_obj = self.ts.get_box(ann_token)
                if box_obj is None: continue
                
                gt_det_dict = {
                    "sample_token": sample_token, 
                    "translation": list(box_obj.center),
                    "size": list(box_obj.wlh),
                    "rotation": list(box_obj.orientation.elements), 
                    "velocity": list(np.nan_to_num(self.ts.box_velocity(ann_token)[:2], nan=0.0)), 
                    "num_pts": ann_record.get('num_lidar_pts', 0) + ann_record.get('num_radar_pts', 0),
                    "detection_name": category_record['name'], 
                    "detection_score": -1.0, 
                    "attribute_name": attribute_records_dict[ann_token][0]['name'] if ann_token in attribute_records_dict and attribute_records_dict[ann_token] else "", 
                    "box_7d": np.array([
                        box_obj.center[0], box_obj.center[1], box_obj.center[2],
                        box_obj.wlh[0], box_obj.wlh[1], box_obj.wlh[2],
                        box_obj.orientation.yaw_pitch_roll[0] 
                    ], dtype=np.float32),
                    "class_idx": class_label_idx,
                    "instance_token": ann_record['instance_token'],
                    "annotation_token": ann_token
                }
                gt_detections_list.append(gt_det_dict)
            except Exception as e:
                if self.verbose: print(f"Warning: Error processing annotation {ann_token} in sample {sample_token}: {e}")
                continue
        return gt_detections_list, annotation_records_dict, instance_records_dict, attribute_records_dict, visibility_records_dict

    def _create_encoder_features(self, box_7d: np.ndarray, is_current: bool, history_slot: Optional[int] = None) -> List[float]:
        features = np.zeros(self.num_encoder_input_features, dtype=np.float32)
        features[:7] = box_7d 

        if 7 < self.num_encoder_input_features: # Index 7 für is_current_flag
            features[7] = 1.0 if is_current else 0.0
        
        if not is_current and history_slot is not None:
            # history_slot: 0 für t-1 (jüngste History), 1 für t-2, ...
            # Feature-Indizes für History-Flags beginnen bei 8
            feature_idx_for_hist_slot = 8 + history_slot 
            if feature_idx_for_hist_slot < self.num_encoder_input_features:
                features[feature_idx_for_hist_slot] = 1.0
        
        return features.tolist()


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_token = self.sample_tokens[idx]
        sample_record = self.ts.get('sample', sample_token)
        
        scene_record = self.ts.get('scene', sample_record['scene_token'])
        scene_meta = parse_scene_description(scene_record['description'])

        sample_data_records: Dict[str, Dict] = {}
        calibrated_sensor_records: Dict[str, Dict] = {}
        for channel, sd_token in sample_record['data'].items():
            sd_rec = self.ts.get('sample_data', sd_token)
            if sd_rec: 
                sample_data_records[channel] = sd_rec
                cs_rec = self.ts.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
                if cs_rec: 
                    calibrated_sensor_records[channel] = cs_rec
        
        ego_pose_record = self.ts.getclosest('ego_pose', sample_record['timestamp'])
        ego_translation_world = np.array(ego_pose_record['translation'], dtype=np.float32)
        ego_rotation_world_quat = np.array(ego_pose_record['rotation'], dtype=np.float32)

        gt_detections_current_frame, ann_recs_curr, inst_recs_curr, attr_recs_curr, vis_recs_curr = \
            self._get_full_box_data_for_token(sample_token)

        combined_encoder_input_features_list: List[List[float]] = []
        combined_encoder_input_xyz_centers_list: List[List[float]] = []

        for gt_item in gt_detections_current_frame:
            box_7d = gt_item['box_7d']
            features = self._create_encoder_features(box_7d, is_current=True, history_slot=None)
            combined_encoder_input_features_list.append(features)
            combined_encoder_input_xyz_centers_list.append(list(box_7d[:3])) 

        history_boxes_7d_raw_list_of_lists: List[List[np.ndarray]] = [] 
        history_sample_tokens_list: List[Optional[str]] = [] 

        prev_token_iterator = sample_record['prev']
        for i in range(self.history_window): 
            history_sample_tokens_list.append(prev_token_iterator) 
            if not prev_token_iterator:
                history_boxes_7d_raw_list_of_lists.append([]) 
                continue

            hist_sample_rec = self.ts.get('sample', prev_token_iterator)
            if not hist_sample_rec:
                history_boxes_7d_raw_list_of_lists.append([])
                break 
            
            hist_frame_gt_detections, _, _, _, _ = self._get_full_box_data_for_token(prev_token_iterator)
            
            current_hist_frame_boxes_7d_list: List[np.ndarray] = []
            for hist_gt_item in hist_frame_gt_detections:
                box_7d = hist_gt_item['box_7d']
                current_hist_frame_boxes_7d_list.append(box_7d)
                
                hist_features = self._create_encoder_features(box_7d, is_current=False, history_slot=i)
                combined_encoder_input_features_list.append(hist_features)
                combined_encoder_input_xyz_centers_list.append(list(box_7d[:3]))
            
            history_boxes_7d_raw_list_of_lists.append(current_hist_frame_boxes_7d_list)
            prev_token_iterator = hist_sample_rec['prev']
            
        history_boxes_7d_raw_list_of_lists.reverse() 
        history_sample_tokens_list.reverse()

        final_encoder_input_features = np.array(combined_encoder_input_features_list, dtype=np.float32) \
            if combined_encoder_input_features_list \
            else np.zeros((0, self.num_encoder_input_features), dtype=np.float32)
        
        final_encoder_input_xyz_centers = np.array(combined_encoder_input_xyz_centers_list, dtype=np.float32) \
            if combined_encoder_input_xyz_centers_list \
            else np.zeros((0, 3), dtype=np.float32)

        processed_history_boxes_7d: List[Optional[np.ndarray]] = []
        for frame_boxes_list in history_boxes_7d_raw_list_of_lists: # frame_boxes_list ist List[np.ndarray]
            if not frame_boxes_list: 
                processed_history_boxes_7d.append(np.zeros((0,7), dtype=np.float32))
            else: # Enthält eine Liste von 7D-Boxen (als np.ndarray)
                processed_history_boxes_7d.append(np.array(frame_boxes_list, dtype=np.float32))
        
        history_padding_mask = np.array([
            (h_boxes is not None and h_boxes.shape[0] > 0)
            for h_boxes in processed_history_boxes_7d 
        ], dtype=bool)


        output = {
            "sample_token": sample_token,
            "timestamp": np.float64(sample_record['timestamp']),
            "prev_sample_token": sample_record['prev'] if sample_record['prev'] else "", 
            "next_sample_token": sample_record['next'] if sample_record['next'] else "", 
            "scene_token": sample_record['scene_token'],
            
            "ego_translation_world": ego_translation_world,
            "ego_rotation_world_quat": ego_rotation_world_quat,
            
            "sample_data_records": sample_data_records,
            "calibrated_sensor_records": calibrated_sensor_records,
            
            "annotation_records_dict": dict(ann_recs_curr), 
            "instance_records_dict": inst_recs_curr,
            "attribute_records_dict": dict(attr_recs_curr), 
            "visibility_records_dict": vis_recs_curr,
            
            "scene_meta": scene_meta,
            
            "gt_detections": gt_detections_current_frame, 
            
            "current_boxes_7d": np.array([item['box_7d'] for item in gt_detections_current_frame], dtype=np.float32) if gt_detections_current_frame else np.zeros((0,7), dtype=np.float32),
            "current_class_labels": np.array([item['class_idx'] for item in gt_detections_current_frame], dtype=np.int64) if gt_detections_current_frame else np.zeros((0,), dtype=np.int64),
            "current_instance_tokens": [item['instance_token'] for item in gt_detections_current_frame],
            "current_annotation_tokens": [item['annotation_token'] for item in gt_detections_current_frame],

            "history_boxes_7d": processed_history_boxes_7d, 
            "history_sample_tokens": history_sample_tokens_list, 
            "history_padding_mask": history_padding_mask, 
            
            "encoder_input_features_for_padding": final_encoder_input_features,
            "encoder_input_xyz_centers_for_padding": final_encoder_input_xyz_centers,
        }
        
        return output

if __name__ == '__main__':
    print("Running ObjectFusionGTDataset example...")
    
    config_file_path = "config/pipeline_c_modules.yaml" 
    if not os.path.exists(config_file_path):
        alt_config_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'pipeline_c_modules.yaml')
        if os.path.exists(alt_config_path):
            config_file_path = alt_config_path
        else:
            print(f"ERROR: Config file not found at {config_file_path} or {alt_config_path}")
            # Erstelle eine minimale Fallback-Config, wenn keine Datei gefunden wird
            print("Using minimal fallback configuration for testing as config file was not found.")
            full_pipeline_config_dict = { 
                "dataset": {"dataroot": "/data", "version": "v1.0-mini", "history_window_transformer": 2, 
                            "class_names": ["car", "truck"]}, 
                "model": {"num_input_features": 10} 
            }
            # exit() # Beende nicht, sondern fahre mit Fallback fort

    if os.path.exists(config_file_path): # Nur laden, wenn Pfad existiert
        try:
            full_pipeline_config_dict = load_config(config_file_path) 
            print(f"Configuration loaded from: {os.path.abspath(config_file_path)}")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            full_pipeline_config_dict = { 
                "dataset": {"dataroot": "/data", "version": "v1.0-mini", "history_window_transformer": 2, 
                            "class_names": ["car", "truck"]}, 
                "model": {"num_input_features": 10} 
            }
            print(f"Using minimal fallback configuration due to loading error: {full_pipeline_config_dict}")
    else: # Fallback, wenn config_file_path nach Prüfung immer noch nicht existiert
        print(f"ERROR: Config file still not found at {config_file_path}. Using minimal fallback.")
        full_pipeline_config_dict = { 
            "dataset": {"dataroot": "/data", "version": "v1.0-mini", "history_window_transformer": 2, 
                        "class_names": ["car", "truck"]}, 
            "model": {"num_input_features": 10} 
        }


    dataset_cfg_from_yaml = full_pipeline_config_dict.get('dataset', {})
    dataroot_path = dataset_cfg_from_yaml.get("dataroot", "/app/datasets") 
    dataset_version = dataset_cfg_from_yaml.get("version", "v1.0-mini")
    
    actual_dataset_path_to_check = os.path.join(dataroot_path, dataset_version)
    if not os.path.exists(actual_dataset_path_to_check):
         print(f"ERROR: Dataset path '{actual_dataset_path_to_check}' does not exist.")
         print(f"  Dataroot from config/fallback: '{dataroot_path}'")
         print(f"  Version from config/fallback: '{dataset_version}'")
         print("  Please ensure 'dataset.dataroot' and 'dataset.version' in your YAML point to the correct TruckScenes directory structure, or that the fallback path is valid in your environment.")
    else:
        try:
            dataset = ObjectFusionGTDataset(
                dataroot=dataroot_path,
                version=dataset_version,
                split_name='mini_val', 
                pipeline_config=full_pipeline_config_dict, 
                verbose=True
            )

            if len(dataset) > 0:
                print(f"\nDataset initialized successfully. Number of samples in '{dataset.split_name}': {len(dataset)}")
                
                num_samples_to_test = min(3, len(dataset))
                print(f"\nTesting first {num_samples_to_test} samples...")
                for i in range(num_samples_to_test): 
                    print(f"\n--- Retrieving Sample {i} ---")
                    sample_data = dataset[i] 
                    print(f"--- Sample {i} (Token: {sample_data['sample_token']}) ---")
                    print(f"  All Keys in output: {list(sample_data.keys())}")
                    
                    assert 'encoder_input_features_for_padding' in sample_data, "ERROR: 'encoder_input_features_for_padding' missing!"
                    features_arr = sample_data['encoder_input_features_for_padding']
                    print(f"  encoder_input_features_for_padding: shape {features_arr.shape}, dtype {features_arr.dtype}")
                    if features_arr.shape[0] > 0 :
                        assert features_arr.shape[1] == dataset.num_encoder_input_features, \
                            f"Feature dimension is {features_arr.shape[1]}, expected {dataset.num_encoder_input_features}"

                    assert 'encoder_input_xyz_centers_for_padding' in sample_data, "ERROR: 'encoder_input_xyz_centers_for_padding' missing!"
                    xyz_arr = sample_data['encoder_input_xyz_centers_for_padding']
                    print(f"  encoder_input_xyz_centers_for_padding: shape {xyz_arr.shape}, dtype {xyz_arr.dtype}")
                    if xyz_arr.shape[0] > 0:
                         assert xyz_arr.shape[1] == 3, f"XYZ dimension is {xyz_arr.shape[1]}, expected 3"
                    
                    assert features_arr.shape[0] == xyz_arr.shape[0], \
                        f"Number of features ({features_arr.shape[0]}) != Number of XYZ centers ({xyz_arr.shape[0]})"

                    if features_arr.shape[0] > 0:
                        print(f"    Example Feature Vector (first object): {features_arr[0].tolist()}")
                        print(f"    Example XYZ Center (first object): {xyz_arr[0].tolist()}")
                    else:
                        print("    No objects (current or history) to form encoder input for this sample.")
                    
                    print(f"  gt_detections (for decoder target): {len(sample_data['gt_detections'])} items")
                    if sample_data['gt_detections']:
                        print(f"    First gt_detection keys: {list(sample_data['gt_detections'][0].keys())}")
                    
                    assert "history_boxes_7d" in sample_data
                    assert "history_sample_tokens" in sample_data
                    assert "history_padding_mask" in sample_data
                    print(f"  history_boxes_7d: {len(sample_data['history_boxes_7d'])} frames")
                    print(f"  history_sample_tokens: {len(sample_data['history_sample_tokens'])} tokens")
                    print(f"  history_padding_mask: {sample_data['history_padding_mask'].tolist()}")

                if len(dataset) >=2: 
                    print("\n--- Testing Collate Function (using the updated dataset output) ---")
                    from oft.data.transformer_gt_collate import object_fusion_gt_collate_fn 
                    
                    num_collate_test_samples = min(4, len(dataset)) 
                    dummy_batch_list = [dataset[k] for k in range(num_collate_test_samples)]
                    
                    print(f"Collating a batch of size {len(dummy_batch_list)}...")
                    collated = object_fusion_gt_collate_fn(dummy_batch_list)
                    print("Collate function returned:")
                    for k, v_tensor in collated.items(): 
                        if torch.is_tensor(v_tensor):
                            print(f"  {k}: Tensor, shape {v_tensor.shape}, dtype {v_tensor.dtype}")
                        elif isinstance(v_tensor, list) and v_tensor and isinstance(v_tensor[0], dict):
                             print(f"  {k}: List of {len(v_tensor)} dicts")
                        elif isinstance(v_tensor, list):
                             print(f"  {k}: List of {len(v_tensor)} items")
                        else:
                            print(f"  {k}: {type(v_tensor)}")
                    
                    assert 'encoder_input_features' in collated, "Collate ERROR: 'encoder_input_features' missing"
                    assert collated['encoder_input_features'].ndim == 3, "Collate ERROR: 'encoder_input_features' falsche Dimensionen"
                    assert 'encoder_input_xyz_centers' in collated, "Collate ERROR: 'encoder_input_xyz_centers' missing"
                    assert collated['encoder_input_xyz_centers'].ndim == 3, "Collate ERROR: 'encoder_input_xyz_centers' falsche Dimensionen"
                    assert collated['encoder_input_xyz_centers'].shape[-1] == 3, "Collate ERROR: 'encoder_input_xyz_centers' letzte Dimension nicht 3"
                    print("Collate function test passed basic key and shape checks for encoder inputs.")

            else:
                print(f"Dataset could be initialized, but no samples were found for split '{dataset.split_name}'.")
        except FileNotFoundError as e:
            print(f"\nError during dataset initialization (FileNotFound): {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred during dataset example: {e}")
            import traceback
            traceback.print_exc()
