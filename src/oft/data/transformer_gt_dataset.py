# src/oft/data/transformer_gt_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from pyquaternion import Quaternion as PyQuaternion

from truckscenes import TruckScenes
from truckscenes.utils.splits import create_splits_scenes
from truckscenes.eval.detection.utils import category_to_detection_name
from truckscenes.eval.detection.constants import DETECTION_NAMES as TRUCKSCENES_DETECTION_NAMES

from oft.utils.common_utils import parse_scene_description
from oft.utils.config import load_config
import os
from omegaconf import OmegaConf, ListConfig, DictConfig


class ObjectFusionGTDataset(Dataset):
    """
    Dataset-Klasse zum Laden von Ground-Truth (GT) 3D-Objektdaten aus TruckScenes.
    Bereitet Daten für ein Transformer-Modell vor.
    - GT-Boxen werden für das Training in FAHRZEUGKOORDINATEN des aktuellen Frames transformiert.
    - Für den Verlust werden die Dimensionen (w,l,h) der GT-Boxen LOGARITHMIERT ('gt_detections_log_dims').
    - Für Matching/GIoU werden GT-Boxen mit TATSÄCHLICHEN Dimensionen bereitgestellt ('gt_detections_actual_dims').
    - Encoder-Input-Features basieren auf TATSÄCHLICHEN Dimensionen in Fahrzeugkoordinaten.
    - Rohe Weltkoordinaten-GTs werden für die Evaluation beibehalten.
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
        if isinstance(pipeline_config, DictConfig):
            self.pipeline_cfg = OmegaConf.to_container(pipeline_config, resolve=True)
        else:
            self.pipeline_cfg = pipeline_config
        self.verbose = verbose

        dataset_specific_cfg = self.pipeline_cfg.get('dataset', {})
        self.history_window = int(dataset_specific_cfg.get('history_window_transformer', 0))

        class_names_from_cfg = dataset_specific_cfg.get('class_names', [])
        if isinstance(class_names_from_cfg, (ListConfig, list, tuple)):
            self.class_names_for_model = list(OmegaConf.to_container(class_names_from_cfg, resolve=True)) \
                                         if isinstance(class_names_from_cfg, ListConfig) else list(class_names_from_cfg)
        else:
            if self.verbose:
                print(f"WARNUNG: dataset.class_names hat unerwarteten Typ {type(class_names_from_cfg)}. "
                      f"Verwende Standard TRUCKSCENES_DETECTION_NAMES.")
            self.class_names_for_model = list(TRUCKSCENES_DETECTION_NAMES)

        model_num_classes = self.pipeline_cfg.get('model', {}).get('num_classes', -1)
        if len(self.class_names_for_model) != model_num_classes and self.verbose:
             print(f"WARNUNG: Anzahl der class_names im Dataset ({len(self.class_names_for_model)}) "
                   f"stimmt nicht mit model.num_classes ({model_num_classes}) überein! "
                   f"Bitte Konfiguration prüfen. Das Modell erwartet {model_num_classes} Klassen.")

        model_specific_cfg = self.pipeline_cfg.get('model', {})
        self.num_encoder_input_features = int(model_specific_cfg.get('num_input_features', 10))

        if self.verbose:
            print(f"Initializing ObjectFusionGTDataset for split '{self.split_name}' with history_window={self.history_window}, num_encoder_input_features={self.num_encoder_input_features}...")
            print(f"  Modell wird auf folgende {len(self.class_names_for_model)} Klassen trainiert: {self.class_names_for_model}")
            print(f"  Box-Koordinaten für Training werden in FAHRZEUGKOORDINATEN des aktuellen Frames transformiert.")
            print(f"  Dimensionen (w,l,h) der GT-Zielboxen für den L1-Dimensionsverlust werden LOGARITHMIERT.")
            print(f"  Encoder-Input-Features basieren auf TATSÄCHLICHEN Dimensionen im Fahrzeugkoordinatensystem.")


        self.ts = TruckScenes(version=self.version, dataroot=self.dataroot, verbose=False)
        scene_name_splits = create_splits_scenes()
        if self.split_name not in scene_name_splits:
            raise ValueError(f"Split '{self.split_name}' not found in TruckScenes splits. Available: {list(scene_name_splits.keys())}")
        self.scene_names_in_split = scene_name_splits[self.split_name]
        self.sample_tokens: List[str] = []
        for scene_record_init in self.ts.scene:
            if scene_record_init['name'] in self.scene_names_in_split:
                current_sample_token = scene_record_init['first_sample_token']
                while current_sample_token:
                    self.sample_tokens.append(current_sample_token)
                    sample_record_check = self.ts.get('sample', current_sample_token)
                    if not sample_record_check or not sample_record_check['next']:
                        break
                    current_sample_token = sample_record_check['next']
        if not self.sample_tokens and self.verbose:
            print(f"WARNING: No sample tokens found for split '{self.split_name}'. Please check dataset and split configuration.")
        elif self.verbose:
            print(f"Collected {len(self.sample_tokens)} sample tokens for split '{self.split_name}'.")

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def _transform_world_to_vehicle(self, world_box_7d: np.ndarray,
                                   current_ego_translation_glob_np: np.ndarray,
                                   current_ego_rotation_glob_pyquat: PyQuaternion) -> np.ndarray:
        box_center_world = world_box_7d[:3]
        box_dims = world_box_7d[3:6]
        box_yaw_world = world_box_7d[6]
        box_center_ego_translated = box_center_world - current_ego_translation_glob_np
        box_center_vehicle = current_ego_rotation_glob_pyquat.inverse.rotate(box_center_ego_translated)
        ego_yaw_world = current_ego_rotation_glob_pyquat.yaw_pitch_roll[0]
        box_yaw_vehicle = box_yaw_world - ego_yaw_world
        box_yaw_vehicle = (box_yaw_vehicle + np.pi) % (2 * np.pi) - np.pi
        return np.array([
            box_center_vehicle[0], box_center_vehicle[1], box_center_vehicle[2],
            box_dims[0], box_dims[1], box_dims[2],
            box_yaw_vehicle
        ], dtype=np.float32)

    def _get_full_box_data_for_token_world(self, sample_token: str) -> List[Dict[str, Any]]:
        gt_detections_list_world: List[Dict[str, Any]] = []
        sample_record = self.ts.get('sample', sample_token)
        if not sample_record: return gt_detections_list_world
        annotation_tokens = sample_record.get('anns', [])
        for ann_token in annotation_tokens:
            try:
                ann_record = self.ts.get('sample_annotation', ann_token)
                if not ann_record: continue
                instance_token = ann_record['instance_token']
                instance_record = self.ts.get('instance', instance_token)
                if not instance_record: continue
                category_record = self.ts.get('category', instance_record['category_token'])
                if not category_record: continue
                original_category_name = category_record['name']
                detection_name_for_eval = category_to_detection_name(original_category_name)
                if detection_name_for_eval is None: continue
                try:
                    class_label_idx = self.class_names_for_model.index(detection_name_for_eval)
                except ValueError: continue
                box_obj = self.ts.get_box(ann_token)
                if box_obj is None: continue
                gt_det_dict_world = {
                    "sample_token": sample_token,
                    "translation_world": list(box_obj.center), "size_world": list(box_obj.wlh),
                    "rotation_world_quat_elements": list(box_obj.orientation.elements),
                    "velocity_world": list(np.nan_to_num(self.ts.box_velocity(ann_token)[:2], nan=0.0)),
                    "num_pts": ann_record.get('num_lidar_pts', 0) + ann_record.get('num_radar_pts', 0),
                    "detection_name": detection_name_for_eval, "detection_score": -1.0,
                    "attribute_name": self.ts.get('attribute', ann_record['attribute_tokens'][0])['name'] if ann_record['attribute_tokens'] else "",
                    "box_7d_world": np.array([box_obj.center[0], box_obj.center[1], box_obj.center[2],
                                              box_obj.wlh[0], box_obj.wlh[1], box_obj.wlh[2],
                                              box_obj.orientation.yaw_pitch_roll[0]], dtype=np.float32),
                    "class_idx": class_label_idx, "instance_token": ann_record['instance_token'],
                    "annotation_token": ann_token, "original_category_name": original_category_name
                }
                gt_detections_list_world.append(gt_det_dict_world)
            except Exception as e:
                if self.verbose: print(f"Dataset Warning: Error processing annotation {ann_token} in sample {sample_token} (world data): {e}")
                continue
        return gt_detections_list_world

    def _create_encoder_features(self, box_7d_vehicle_actual_dims: np.ndarray, is_current: bool, history_slot: Optional[int] = None) -> List[float]:
        features = np.zeros(self.num_encoder_input_features, dtype=np.float32)
        if not isinstance(box_7d_vehicle_actual_dims, np.ndarray) or box_7d_vehicle_actual_dims.shape[0] != 7:
            return features.tolist()
        features[:7] = box_7d_vehicle_actual_dims
        if 7 < self.num_encoder_input_features:
            features[7] = 1.0 if is_current else 0.0
        if not is_current and history_slot is not None:
            feature_idx_for_hist_slot = 8 + history_slot
            if feature_idx_for_hist_slot < self.num_encoder_input_features:
                features[feature_idx_for_hist_slot] = 1.0
        return features.tolist()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_token = self.sample_tokens[idx]
        sample_record = self.ts.get('sample', sample_token)
        scene_record = self.ts.get('scene', sample_record['scene_token'])
        ego_pose_record = self.ts.getclosest('ego_pose', sample_record['timestamp'])
        ego_translation_world_np = np.array(ego_pose_record['translation'], dtype=np.float32)
        ego_rotation_world_pyquat = PyQuaternion(ego_pose_record['rotation'])

        gt_detections_raw_world_current_frame: List[Dict[str, Any]] = self._get_full_box_data_for_token_world(sample_token)
        
        gt_detections_current_frame_vehicle_logdims: List[Dict[str, Any]] = []
        gt_detections_current_frame_vehicle_actualdims: List[Dict[str, Any]] = [] # NEU

        for gt_item_world in gt_detections_raw_world_current_frame:
            box_7d_world = gt_item_world['box_7d_world']
            box_7d_vehicle_actual = self._transform_world_to_vehicle(box_7d_world, ego_translation_world_np, ego_rotation_world_pyquat)
            
            # Version mit log-Dimensionen für Targets
            box_7d_vehicle_log = box_7d_vehicle_actual.copy()
            box_7d_vehicle_log[3:6] = np.log(np.maximum(box_7d_vehicle_actual[3:6], 1e-5))

            gt_item_vehicle_logdims = gt_item_world.copy()
            gt_item_vehicle_logdims['box_7d'] = box_7d_vehicle_log
            gt_item_vehicle_logdims['translation'] = box_7d_vehicle_log[:3].tolist()
            new_quat_vehicle_logdims = PyQuaternion(axis=[0,0,1], radians=box_7d_vehicle_log[6])
            gt_item_vehicle_logdims['rotation'] = list(new_quat_vehicle_logdims.elements)
            gt_detections_current_frame_vehicle_logdims.append(gt_item_vehicle_logdims)

            # Version mit aktuellen Dimensionen für Matching/GIoU und Encoder-Features
            gt_item_vehicle_actualdims = gt_item_world.copy()
            gt_item_vehicle_actualdims['box_7d'] = box_7d_vehicle_actual # Behält aktuelle Dimensionen
            gt_item_vehicle_actualdims['translation'] = box_7d_vehicle_actual[:3].tolist()
            new_quat_vehicle_actualdims = PyQuaternion(axis=[0,0,1], radians=box_7d_vehicle_actual[6])
            gt_item_vehicle_actualdims['rotation'] = list(new_quat_vehicle_actualdims.elements)
            gt_detections_current_frame_vehicle_actualdims.append(gt_item_vehicle_actualdims)


        current_class_labels_np = np.array([item['class_idx'] for item in gt_detections_current_frame_vehicle_logdims], dtype=np.int64) \
            if gt_detections_current_frame_vehicle_logdims else np.zeros((0,), dtype=np.int64)

        combined_encoder_input_features_list_vehicle: List[List[float]] = []
        combined_encoder_input_xyz_centers_list_vehicle: List[List[float]] = []
        
        # Encoder-Features basieren auf Boxen mit TATSÄCHLICHEN Dimensionen
        for gt_item_veh_actual in gt_detections_current_frame_vehicle_actualdims:
            box_7d_veh_actual = gt_item_veh_actual['box_7d'] # Dies sind die tatsächlichen Fahrzeug-Dims
            features_veh = self._create_encoder_features(box_7d_veh_actual, is_current=True, history_slot=None)
            combined_encoder_input_features_list_vehicle.append(features_veh)
            combined_encoder_input_xyz_centers_list_vehicle.append(list(box_7d_veh_actual[:3]))

        temp_history_vehicle_frames_actual_dims: List[List[np.ndarray]] = []
        temp_history_tokens_raw: List[Optional[str]] = []
        prev_token_iterator = sample_record['prev']

        for _ in range(self.history_window):
            if not prev_token_iterator:
                temp_history_vehicle_frames_actual_dims.append([])
                temp_history_tokens_raw.append(None)
                continue
            hist_sample_rec = self.ts.get('sample', prev_token_iterator)
            temp_history_tokens_raw.append(prev_token_iterator)
            if not hist_sample_rec:
                temp_history_vehicle_frames_actual_dims.append([])
                prev_token_iterator = None
                continue
            hist_frame_gt_detections_world = self._get_full_box_data_for_token_world(prev_token_iterator)
            current_hist_frame_boxes_7d_vehicle_actual_list: List[np.ndarray] = []
            for hist_gt_item_world in hist_frame_gt_detections_world:
                box_7d_world_hist = hist_gt_item_world['box_7d_world']
                box_7d_vehicle_hist_actual = self._transform_world_to_vehicle(box_7d_world_hist, ego_translation_world_np, ego_rotation_world_pyquat)
                current_hist_frame_boxes_7d_vehicle_actual_list.append(box_7d_vehicle_hist_actual)
            temp_history_vehicle_frames_actual_dims.append(current_hist_frame_boxes_7d_vehicle_actual_list)
            if hist_sample_rec: prev_token_iterator = hist_sample_rec['prev']
            else: prev_token_iterator = None

        for history_slot_idx, hist_frame_boxes_list_veh_actual in enumerate(temp_history_vehicle_frames_actual_dims):
            for box_7d_veh_hist_actual in hist_frame_boxes_list_veh_actual: # Dies sind Boxen mit aktuellen Dims
                hist_features_veh = self._create_encoder_features(box_7d_veh_hist_actual, is_current=False, history_slot=history_slot_idx)
                combined_encoder_input_features_list_vehicle.append(hist_features_veh)
                combined_encoder_input_xyz_centers_list_vehicle.append(list(box_7d_veh_hist_actual[:3]))
        
        history_for_output_dict_boxes_veh_actual_dims = [np.array(frame_boxes, dtype=np.float32) if frame_boxes else np.zeros((0,7), dtype=np.float32) for frame_boxes in reversed(temp_history_vehicle_frames_actual_dims)]
        history_for_output_dict_tokens = list(reversed(temp_history_tokens_raw))
        
        history_padding_mask_np = np.array([
            (isinstance(h_boxes, np.ndarray) and h_boxes.shape[0] > 0)
            for h_boxes in history_for_output_dict_boxes_veh_actual_dims
        ], dtype=bool) if self.history_window > 0 else np.array([], dtype=bool)

        final_encoder_input_features_vehicle = np.array(combined_encoder_input_features_list_vehicle, dtype=np.float32) \
            if combined_encoder_input_features_list_vehicle \
            else np.zeros((0, self.num_encoder_input_features), dtype=np.float32)
        final_encoder_input_xyz_centers_vehicle = np.array(combined_encoder_input_xyz_centers_list_vehicle, dtype=np.float32) \
            if combined_encoder_input_xyz_centers_list_vehicle \
            else np.zeros((0, 3), dtype=np.float32)
            
        output = {
            "sample_token": sample_token,
            "timestamp": np.float64(sample_record['timestamp']),
            "ego_translation_world": ego_translation_world_np,
            "ego_rotation_world_quat": np.array(ego_rotation_world_pyquat.elements, dtype=np.float32),
            
            "gt_detections_log_dims": gt_detections_current_frame_vehicle_logdims, 
            "gt_detections_actual_dims": gt_detections_current_frame_vehicle_actualdims, # NEU
            "gt_detections_raw_world_current_frame": gt_detections_raw_world_current_frame,

            "encoder_input_features_for_padding": final_encoder_input_features_vehicle,
            "encoder_input_xyz_centers_for_padding": final_encoder_input_xyz_centers_vehicle,

            "history_boxes_7d_actual_dims": history_for_output_dict_boxes_veh_actual_dims,
            "history_sample_tokens": history_for_output_dict_tokens,
            "history_padding_mask": history_padding_mask_np,
            
            "current_class_labels": current_class_labels_np,
            "scene_meta": parse_scene_description(scene_record['description']) if scene_record else {"error": "scene_record not found"}
        }
        return output

if __name__ == '__main__':
    print("Starte Test für ObjectFusionGTDataset (Fahrzeugkoordinaten & Log/Actual-Dimensionen für Targets)...")
    
    default_config_path_main = "config/pipeline_c_modules.yaml"
    project_root_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    config_file_path_main = os.path.join(project_root_main, default_config_path_main)

    pipeline_cfg_dict_main = None
    if not os.path.exists(config_file_path_main):
        print(f"FEHLER: Konfigurationsdatei nicht gefunden: {config_file_path_main}")
        from truckscenes.eval.detection.constants import DETECTION_NAMES as TRUCKSCENES_DET_NAMES_MAIN
        pipeline_cfg_dict_main = {
            "dataset": {"dataroot": "/data", "version": "v1.0-mini", 
                        "history_window_transformer": 2, "class_names": list(TRUCKSCENES_DET_NAMES_MAIN)},
            "model": {"num_input_features": 10, "num_classes": 12, "pe_max_coord_val": 150.0}
        }
        print("Verwende Fallback-Konfiguration für Dataset-Test.")
    else:
        pipeline_cfg_dict_main = load_config(config_file_path_main)
        if not isinstance(pipeline_cfg_dict_main, dict):
             pipeline_cfg_dict_main = OmegaConf.to_container(pipeline_cfg_dict_main, resolve=True)
        print(f"Konfiguration geladen von: {config_file_path_main}")

    dataroot_main = pipeline_cfg_dict_main.get('dataset',{}).get("dataroot", "/data")
    if dataroot_main == "/data" and not os.path.exists("/data/v1.0-mini"):
        dataroot_main = os.path.join(project_root_main, "data", "sets", "truckscenes")
        print(f"Lokaler Test: /data/v1.0-mini nicht gefunden, verwende stattdessen: {dataroot_main}")
        dummy_version_path = os.path.join(dataroot_main, pipeline_cfg_dict_main.get('dataset',{}).get("version", "v1.0-mini"))
        if not os.path.exists(dummy_version_path):
            print(f"Erstelle Dummy-Dataset-Struktur unter {dummy_version_path} für den Test...")
            os.makedirs(dummy_version_path, exist_ok=True)
            import json
            for table_name_dummy in ['attribute', 'calibrated_sensor', 'category', 'ego_motion_cabin',
                                     'ego_motion_chassis', 'ego_pose', 'instance', 'sample',
                                     'sample_annotation', 'sample_data', 'scene', 'sensor', 'visibility']:
                with open(os.path.join(dummy_version_path, f"{table_name_dummy}.json"), 'w') as f:
                    json.dump([], f)
            print("Dummy-Dataset-Struktur erstellt.")

    dataset_version_main = pipeline_cfg_dict_main.get('dataset',{}).get("version", "v1.0-mini")
    actual_dataset_path_main = os.path.join(dataroot_main, dataset_version_main)

    if not os.path.exists(actual_dataset_path_main) and not dataroot_main=="/data":
        print(f"FEHLER: Dataset-Pfad '{actual_dataset_path_main}' existiert nicht.")
    else:
        print(f"Versuche Dataset zu laden von: {actual_dataset_path_main}")
        try:
            dataset_instance = ObjectFusionGTDataset(
                dataroot=dataroot_main, version=dataset_version_main,
                split_name='mini_val', pipeline_config=pipeline_cfg_dict_main, verbose=True
            )
            if len(dataset_instance) > 0:
                print(f"\nDataset erfolgreich initialisiert. Samples: {len(dataset_instance)}")
                num_samples_to_show = min(1, len(dataset_instance))
                for i in range(num_samples_to_show):
                    print(f"\n--- Sample {i} (Token: {dataset_instance.sample_tokens[i]}) ---")
                    sample_output = dataset_instance[i]
                    print(f"  Output-Schlüssel: {list(sample_output.keys())}")
                    
                    # GTs für Loss (Fahrzeugkoordinaten, log-dims)
                    if 'gt_detections_log_dims' in sample_output and sample_output['gt_detections_log_dims']:
                        first_gt_logdims = sample_output['gt_detections_log_dims'][0]
                        print(f"  GT für Loss (Fahrzeug, Log-Dims) ['gt_detections_log_dims'][0]:")
                        print(f"    box_7d (log-dims): {np.round(first_gt_logdims['box_7d'], 3)}")
                        original_dims_world = first_gt_logdims.get('size_world', [0,0,0]) # Aus _get_full_box_data_for_token_world
                        reconstructed_dims_from_log = np.exp(first_gt_logdims['box_7d'][3:6])
                        print(f"      Original Dims (world): {np.round(original_dims_world,2)}")
                        print(f"      Log Dims in 'box_7d': {np.round(first_gt_logdims['box_7d'][3:6],3)}")
                        print(f"      Rekonstruierte Dims (exp(log-dims)): {np.round(reconstructed_dims_from_log,2)}")
                    else:
                        print("  'gt_detections_log_dims' ist leer oder nicht vorhanden.")

                    # GTs mit aktuellen Dims (Fahrzeugkoordinaten)
                    if 'gt_detections_actual_dims' in sample_output and sample_output['gt_detections_actual_dims']:
                        first_gt_actualdims = sample_output['gt_detections_actual_dims'][0]
                        print(f"  GT für Matching (Fahrzeug, Aktuelle Dims) ['gt_detections_actual_dims'][0]:")
                        print(f"    box_7d (aktuelle Dims): {np.round(first_gt_actualdims['box_7d'], 3)}")
                    else:
                        print("  'gt_detections_actual_dims' ist leer oder nicht vorhanden.")
                    
                    enc_features = sample_output['encoder_input_features_for_padding']
                    if enc_features.ndim > 1 and enc_features.shape[0] > 0 :
                        print(f"  Encoder Features (erste Zeile, Dims sollten aktuell sein): {np.round(enc_features[0, 3:6], 2)}")
                    else:
                        print(f"  Keine Encoder Features in diesem Sample (Shape: {enc_features.shape}).")
            else: print(f"Keine Samples im Dataset für Split '{dataset_instance.split_name}'.")
        except Exception as e:
            print(f"\nFehler bei Dataset-Test: {e}")
            import traceback; traceback.print_exc()
