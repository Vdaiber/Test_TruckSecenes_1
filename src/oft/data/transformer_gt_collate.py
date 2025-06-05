# src/oft/data/transformer_gt_collate.py
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
import os # für __main__ test

def object_fusion_gt_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate-Funktion für das ObjectFusionGTDataset.
    - 'gt_target_boxes_log_dims' enthält GT-Boxen mit LOGARITHMIERTEN Dimensionen in Fahrzeugkoordinaten (für L1-Dimensionsloss).
    - 'gt_target_boxes_actual_dims' enthält GT-Boxen mit TATSÄCHLICHEN Dimensionen in Fahrzeugkoordinaten (für Matcher/GIoU).
    - 'gt_detections_list_raw' enthält GTs in WELTKOORDINATEN mit TATSÄCHLICHEN Dimensionen (für DevKit Eval).
    """
    
    sample_tokens = [item['sample_token'] for item in batch]
    timestamps_np = np.array([item['timestamp'] for item in batch])
    
    ego_translations_world = torch.tensor(np.array([item['ego_translation_world'] for item in batch]), dtype=torch.float32)
    ego_rotations_world_quat = torch.tensor(np.array([item['ego_rotation_world_quat'] for item in batch]), dtype=torch.float32)

    # Für DevKit Evaluation (Weltkoordinaten, aktuelle Dimensionen)
    gt_detections_list_raw_world_batch = [item['gt_detections_raw_world_current_frame'] for item in batch]

    # Für Loss Targets
    gt_detections_for_loss_targets_logdims = [item['gt_detections_log_dims'] for item in batch]
    # NEU: Für Matching und GIoU (Fahrzeugkoordinaten, aktuelle Dimensionen)
    gt_detections_for_matching_actualdims = [item['gt_detections_actual_dims'] for item in batch]


    encoder_input_features_list = [item['encoder_input_features_for_padding'] for item in batch]
    encoder_input_xyz_centers_list = [item['encoder_input_xyz_centers_for_padding'] for item in batch]

    batch_size = len(batch)
    max_encoder_inputs = 0
    if any(isinstance(features, np.ndarray) and features.shape[0] > 0 for features in encoder_input_features_list):
        max_encoder_inputs = max(features.shape[0] for features in encoder_input_features_list if isinstance(features, np.ndarray) and features.shape[0] > 0)
    
    num_encoder_input_features_dim = 10
    for features_sample_check in encoder_input_features_list:
        if isinstance(features_sample_check, np.ndarray) and features_sample_check.ndim == 2 and features_sample_check.shape[0] > 0:
            num_encoder_input_features_dim = features_sample_check.shape[1]
            break
            
    encoder_input_features_padded = torch.zeros((batch_size, max_encoder_inputs, num_encoder_input_features_dim), dtype=torch.float32)
    encoder_input_xyz_centers_padded = torch.zeros((batch_size, max_encoder_inputs, 3), dtype=torch.float32)
    encoder_input_mask_padded = torch.ones((batch_size, max_encoder_inputs), dtype=torch.bool)

    for i, features_np in enumerate(encoder_input_features_list):
        if not isinstance(features_np, np.ndarray):
            encoder_input_mask_padded[i, :] = True; continue
        xyz_centers_np = encoder_input_xyz_centers_list[i]
        if not isinstance(xyz_centers_np, np.ndarray):
            encoder_input_mask_padded[i, :] = True; continue
        num_current_encoder_inputs = features_np.shape[0]
        if num_current_encoder_inputs > 0:
            if features_np.ndim == 2 and features_np.shape[1] == num_encoder_input_features_dim :
                 encoder_input_features_padded[i, :num_current_encoder_inputs, :] = torch.from_numpy(features_np)
            elif features_np.ndim == 2 :
                 encoder_input_mask_padded[i, :num_current_encoder_inputs] = True; continue
            else:
                 encoder_input_mask_padded[i, :num_current_encoder_inputs] = True; continue
            if xyz_centers_np.shape[0] == num_current_encoder_inputs and xyz_centers_np.ndim == 2 and xyz_centers_np.shape[1] == 3:
                encoder_input_xyz_centers_padded[i, :num_current_encoder_inputs, :] = torch.from_numpy(xyz_centers_np)
            elif num_current_encoder_inputs > 0 :
                 pass 
            encoder_input_mask_padded[i, :num_current_encoder_inputs] = False

    # --- Ground-Truth Targets für den Decoder vorbereiten ---
    box_dim = 7
    
    # 1. GTs mit LOG-Dimensionen (für L1-Dimensionsloss)
    max_gt_objects_log = 0
    if any(isinstance(sample_gts, list) and len(sample_gts) > 0 for sample_gts in gt_detections_for_loss_targets_logdims):
         max_gt_objects_log = max(len(sample_gts) for sample_gts in gt_detections_for_loss_targets_logdims if isinstance(sample_gts, list) and len(sample_gts) > 0)

    gt_target_boxes_padded_logdims = torch.zeros((batch_size, max_gt_objects_log, box_dim), dtype=torch.float32)
    gt_target_labels_padded_log = torch.full((batch_size, max_gt_objects_log), -1, dtype=torch.long) # Labels sind für beide gleich
    gt_target_valid_mask_log = torch.zeros((batch_size, max_gt_objects_log), dtype=torch.bool)

    for i, sample_gts_list_logdims in enumerate(gt_detections_for_loss_targets_logdims):
        if isinstance(sample_gts_list_logdims, list) and sample_gts_list_logdims:
            current_gt_boxes_logdims_list = []
            current_gt_labels_list = [] # Wird nur einmal pro Sample gesammelt
            for gt_dict_logdims in sample_gts_list_logdims:
                if 'box_7d' in gt_dict_logdims and isinstance(gt_dict_logdims['box_7d'], np.ndarray) and \
                   gt_dict_logdims['box_7d'].shape == (7,) and 'class_idx' in gt_dict_logdims:
                    current_gt_boxes_logdims_list.append(gt_dict_logdims['box_7d'])
                    current_gt_labels_list.append(gt_dict_logdims['class_idx'])
                else: pass
            if current_gt_boxes_logdims_list:
                current_gt_boxes_logdims_np = np.array(current_gt_boxes_logdims_list, dtype=np.float32)
                current_gt_labels_np = np.array(current_gt_labels_list, dtype=np.int64)
                num_current_gts = current_gt_boxes_logdims_np.shape[0]
                
                # Stelle sicher, dass max_gt_objects_log nicht 0 ist, wenn num_current_gts > 0
                if num_current_gts > 0 and max_gt_objects_log == 0: max_gt_objects_log = num_current_gts

                if num_current_gts > 0 and num_current_gts <= max_gt_objects_log:
                    gt_target_boxes_padded_logdims[i, :num_current_gts, :] = torch.from_numpy(current_gt_boxes_logdims_np)
                    gt_target_labels_padded_log[i, :num_current_gts] = torch.from_numpy(current_gt_labels_np)
                    gt_target_valid_mask_log[i, :num_current_gts] = True
                elif num_current_gts > max_gt_objects_log:
                    gt_target_boxes_padded_logdims[i, :, :] = torch.from_numpy(current_gt_boxes_logdims_np[:max_gt_objects_log])
                    gt_target_labels_padded_log[i, :] = torch.from_numpy(current_gt_labels_np[:max_gt_objects_log])
                    gt_target_valid_mask_log[i, :] = True
    
    # 2. GTs mit TATSÄCHLICHEN Dimensionen (für Matcher/GIoU)
    max_gt_objects_actual = 0
    if any(isinstance(sample_gts, list) and len(sample_gts) > 0 for sample_gts in gt_detections_for_matching_actualdims):
         max_gt_objects_actual = max(len(sample_gts) for sample_gts in gt_detections_for_matching_actualdims if isinstance(sample_gts, list) and len(sample_gts) > 0)

    gt_target_boxes_padded_actualdims = torch.zeros((batch_size, max_gt_objects_actual, box_dim), dtype=torch.float32)
    # Labels und Maske sollten identisch sein, da die Anzahl der Objekte gleich ist
    gt_target_labels_padded_actual = torch.full((batch_size, max_gt_objects_actual), -1, dtype=torch.long)
    gt_target_valid_mask_actual = torch.zeros((batch_size, max_gt_objects_actual), dtype=torch.bool)

    for i, sample_gts_list_actualdims in enumerate(gt_detections_for_matching_actualdims):
        if isinstance(sample_gts_list_actualdims, list) and sample_gts_list_actualdims:
            current_gt_boxes_actualdims_list = []
            current_gt_labels_list_actual = [] # Labels erneut sammeln für Konsistenz
            for gt_dict_actualdims in sample_gts_list_actualdims:
                if 'box_7d' in gt_dict_actualdims and isinstance(gt_dict_actualdims['box_7d'], np.ndarray) and \
                   gt_dict_actualdims['box_7d'].shape == (7,) and 'class_idx' in gt_dict_actualdims:
                    current_gt_boxes_actualdims_list.append(gt_dict_actualdims['box_7d'])
                    current_gt_labels_list_actual.append(gt_dict_actualdims['class_idx'])
                else: pass
            if current_gt_boxes_actualdims_list:
                current_gt_boxes_actualdims_np = np.array(current_gt_boxes_actualdims_list, dtype=np.float32)
                current_gt_labels_actual_np = np.array(current_gt_labels_list_actual, dtype=np.int64)
                num_current_gts_actual = current_gt_boxes_actualdims_np.shape[0]

                if num_current_gts_actual > 0 and max_gt_objects_actual == 0: max_gt_objects_actual = num_current_gts_actual

                if num_current_gts_actual > 0 and num_current_gts_actual <= max_gt_objects_actual:
                    gt_target_boxes_padded_actualdims[i, :num_current_gts_actual, :] = torch.from_numpy(current_gt_boxes_actualdims_np)
                    gt_target_labels_padded_actual[i, :num_current_gts_actual] = torch.from_numpy(current_gt_labels_actual_np)
                    gt_target_valid_mask_actual[i, :num_current_gts_actual] = True
                elif num_current_gts_actual > max_gt_objects_actual:
                    gt_target_boxes_padded_actualdims[i, :, :] = torch.from_numpy(current_gt_boxes_actualdims_np[:max_gt_objects_actual])
                    gt_target_labels_padded_actual[i, :] = torch.from_numpy(current_gt_labels_actual_np[:max_gt_objects_actual])
                    gt_target_valid_mask_actual[i, :] = True
    
    # Sicherstellen, dass die Anzahl der Objekte und somit die Label/Masken konsistent sind
    # Dies ist eine Vereinfachung; idealerweise würde man die Objekte über Tokens matchen,
    # aber für GT->GT sollte die Reihenfolge und Anzahl identisch sein.
    if not torch.equal(gt_target_labels_padded_log, gt_target_labels_padded_actual):
        # print("WARNUNG CollateFn: Labels für log_dims und actual_dims Targets stimmen nicht überein. Verwende Labels von log_dims.")
        pass # Weniger verbose, für GT->GT sollte es passen.
    if not torch.equal(gt_target_valid_mask_log, gt_target_valid_mask_actual):
        # print("WARNUNG CollateFn: Valid_mask für log_dims und actual_dims Targets stimmen nicht überein. Verwende Maske von log_dims.")
        pass

    collated_batch = {
        'sample_tokens': sample_tokens,
        'timestamps': torch.tensor(timestamps_np, dtype=torch.float64),
        'ego_translations_world': ego_translations_world,
        'ego_rotations_world_quat': ego_rotations_world_quat,
        'encoder_input_features': encoder_input_features_padded,
        'encoder_input_xyz_centers': encoder_input_xyz_centers_padded,
        'encoder_input_mask': encoder_input_mask_padded,
        
        'gt_target_boxes_log_dims': gt_target_boxes_padded_logdims, # Für L1-Dim Loss
        'gt_target_boxes_actual_dims': gt_target_boxes_padded_actualdims, # Für Matcher & GIoU
        'gt_target_labels': gt_target_labels_padded_log, # Labels sollten für beide gleich sein
        'gt_target_valid_mask': gt_target_valid_mask_log, # Valid-Maske sollte auch gleich sein
        
        'gt_detections_list_raw': gt_detections_list_raw_world_batch
    }
    return collated_batch

if __name__ == '__main__':
    print("Starte Test für object_fusion_gt_collate_fn (Log- & Aktuelle Dims)...")
    try:
        default_config_path_collate = "config/pipeline_c_modules.yaml"
        project_root_collate = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        config_file_path_collate = os.path.join(project_root_collate, default_config_path_collate)

        if not os.path.exists(config_file_path_collate):
            print(f"Config file not found at {config_file_path_collate}.")
            from truckscenes.eval.detection.constants import DETECTION_NAMES as TRUCKSCENES_DET_NAMES_COLL_TEST
            full_pipeline_config_dict_collate = {
                "dataset": {"dataroot": "/data", "version": "v1.0-mini", "history_window_transformer": 2, "class_names": list(TRUCKSCENES_DET_NAMES_COLL_TEST)},
                "model": {"num_input_features": 10, "num_classes": 12, "pe_max_coord_val": 150.0}
            }
            print("Verwende Fallback-Konfiguration für Collate-Test.")
        else:
            from oft.utils.config import load_config
            full_pipeline_config_dict_collate = load_config(config_file_path_collate)
            print(f"Konfiguration für CollateFn-Test geladen von: {os.path.abspath(config_file_path_collate)}")
        
        if not isinstance(full_pipeline_config_dict_collate, dict):
            from omegaconf import OmegaConf
            full_pipeline_config_dict_collate = OmegaConf.to_container(full_pipeline_config_dict_collate, resolve=True)

        from oft.data.transformer_gt_dataset import ObjectFusionGTDataset

        dataset_cfg_collate = full_pipeline_config_dict_collate.get('dataset', {})
        dataroot_path_collate = dataset_cfg_collate.get("dataroot", "/data")
        if dataroot_path_collate == "/data" and not os.path.exists("/data/v1.0-mini"):
            dataroot_path_collate = os.path.join(project_root_collate, "data", "sets", "truckscenes")
            print(f"Lokaler Test: Collate - /data nicht gefunden, verwende: {dataroot_path_collate}")
            dummy_version_path_collate = os.path.join(dataroot_path_collate, dataset_cfg_collate.get("version", "v1.0-mini"))
            if not os.path.exists(dummy_version_path_collate):
                os.makedirs(dummy_version_path_collate, exist_ok=True)
                import json
                for table_name_dummy_collate in ['sample', 'scene', 'sample_annotation', 'instance', 'category', 'attribute', 'visibility', 'sensor', 'calibrated_sensor', 'ego_pose']:
                    with open(os.path.join(dummy_version_path_collate, f"{table_name_dummy_collate}.json"), 'w') as f_collate: json.dump([], f_collate)

        dataset_version_collate = dataset_cfg_collate.get("version", "v1.0-mini")
        actual_dataset_path_to_check_collate = os.path.join(dataroot_path_collate, dataset_version_collate)

        if not os.path.exists(actual_dataset_path_to_check_collate) and not dataroot_path_collate == "/data":
             print(f"FEHLER: Dataset-Pfad '{actual_dataset_path_to_check_collate}' für CollateFn-Test nicht gefunden.")
        else:
            print(f"Versuche Dataset für CollateFn zu laden von: {actual_dataset_path_to_check_collate}")
            example_dataset = ObjectFusionGTDataset(
                dataroot=dataroot_path_collate, version=dataset_version_collate,
                split_name="mini_val", pipeline_config=full_pipeline_config_dict_collate, verbose=False
            )
            
            if len(example_dataset) >= 1:
                print(f"Dataset für CollateFn-Test geladen, {len(example_dataset)} Samples im '{example_dataset.split_name}' Split.")
                dummy_batch_list = [example_dataset[0]]
                
                print(f"\nTeste Collate-Funktion mit Batch der Größe {len(dummy_batch_list)}...")
                collated_batch = object_fusion_gt_collate_fn(dummy_batch_list)
                print("Collate-Funktion erfolgreich ausgeführt.")
                
                print("\nÜberprüfe Output-Schlüssel:")
                assert 'gt_target_boxes_log_dims' in collated_batch
                assert 'gt_target_boxes_actual_dims' in collated_batch
                assert 'gt_target_labels' in collated_batch
                assert 'gt_target_valid_mask' in collated_batch

                if collated_batch['gt_target_boxes_log_dims'].numel() > 0:
                    log_box = collated_batch['gt_target_boxes_log_dims'][0,0,:].cpu().numpy()
                    print(f"  Beispiel gt_target_boxes_log_dims[0,0,:]: {np.round(log_box, 3)}")
                    print(f"    Zentrum: {np.round(log_box[:3], 3)}, Log-Dims: {np.round(log_box[3:6], 3)}, Yaw: {log_box[6]:.3f}")
                
                if collated_batch['gt_target_boxes_actual_dims'].numel() > 0:
                    actual_box = collated_batch['gt_target_boxes_actual_dims'][0,0,:].cpu().numpy()
                    print(f"  Beispiel gt_target_boxes_actual_dims[0,0,:]: {np.round(actual_box, 3)}")
                    print(f"    Zentrum: {np.round(actual_box[:3], 3)}, Aktuelle Dims: {np.round(actual_box[3:6], 3)}, Yaw: {actual_box[6]:.3f}")

                print("\nGrundlegende Tests für Collate-Funktion (Log- & Aktuelle Dims) erfolgreich.")
            else:
                print(f"Nicht genügend Samples für CollateFn-Test ({len(example_dataset)} gefunden).")
    except Exception as e:
        print(f"\nEin unerwarteter Fehler ist während des CollateFn-Tests aufgetreten: {e}")
        import traceback
        traceback.print_exc()