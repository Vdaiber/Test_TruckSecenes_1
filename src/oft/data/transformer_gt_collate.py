# src/oft/data/transformer_gt_collate.py
import torch
import numpy as np
from typing import List, Dict, Any, Tuple

def object_fusion_gt_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate-Funktion für das ObjectFusionGTDataset.
    Nimmt einen Batch von Samples und bereitet sie für den Transformer-Input vor.
    Fügt 'gt_detections_list_raw' für die Evaluation hinzu.
    Stellt sicher, dass 'encoder_input_xyz_centers' im Batch enthalten ist.
    """
    
    # --- 1. Sammle alle Datenfelder aus dem Batch ---\
    sample_tokens = [item['sample_token'] for item in batch]
    timestamps_np = np.array([item['timestamp'] for item in batch]) # Erst als NumPy-Array sammeln
    
    ego_translations_world = torch.tensor(np.array([item['ego_translation_world'] for item in batch]), dtype=torch.float32)
    ego_rotations_world_quat = torch.tensor(np.array([item['ego_rotation_world_quat'] for item in batch]), dtype=torch.float32)

    gt_detections_list_raw_batch = [item['gt_detections'] for item in batch]
    gt_detections_for_tensor_processing = [item['gt_detections'] for item in batch] 

    encoder_input_features_list = [item['encoder_input_features_for_padding'] for item in batch]
    encoder_input_xyz_centers_list = [item['encoder_input_xyz_centers_for_padding'] for item in batch]

    # --- 2. Bereite Encoder-Inputs vor (Padding etc.) ---
    batch_size = len(batch)
    
    max_encoder_inputs = 0
    if any(isinstance(features, np.ndarray) and features.shape[0] > 0 for features in encoder_input_features_list):
        max_encoder_inputs = max(features.shape[0] for features in encoder_input_features_list if isinstance(features, np.ndarray) and features.shape[0] > 0)
    
    num_encoder_input_features_dim = 10 
    for features_sample in encoder_input_features_list:
        if isinstance(features_sample, np.ndarray) and features_sample.shape[0] > 0:
            num_encoder_input_features_dim = features_sample.shape[1]
            break
            
    encoder_input_features_padded = torch.zeros((batch_size, max_encoder_inputs, num_encoder_input_features_dim), dtype=torch.float32)
    encoder_input_xyz_centers_padded = torch.zeros((batch_size, max_encoder_inputs, 3), dtype=torch.float32) 
    encoder_input_mask_padded = torch.ones((batch_size, max_encoder_inputs), dtype=torch.bool) 

    for i, features_np in enumerate(encoder_input_features_list):
        # Stelle sicher, dass features_np und xyz_centers_np NumPy-Arrays sind
        if not isinstance(features_np, np.ndarray):
            print(f"WARNUNG: CollateFn - features_np für Sample {i} ist kein np.ndarray, sondern {type(features_np)}. Überspringe Features.")
            continue # Oder behandle als leeres Array

        xyz_centers_np = encoder_input_xyz_centers_list[i]
        if not isinstance(xyz_centers_np, np.ndarray):
            print(f"WARNUNG: CollateFn - xyz_centers_np für Sample {i} ist kein np.ndarray, sondern {type(xyz_centers_np)}. Überspringe XYZ-Zentren.")
            # Setze ggf. die Maske für dieses Sample komplett auf True, wenn Features fehlen
            encoder_input_mask_padded[i, :] = True 
            continue


        num_current_encoder_inputs = features_np.shape[0]
        if num_current_encoder_inputs > 0:
            encoder_input_features_padded[i, :num_current_encoder_inputs, :] = torch.from_numpy(features_np)
            
            if xyz_centers_np.shape[0] == num_current_encoder_inputs and xyz_centers_np.shape[1] == 3:
                encoder_input_xyz_centers_padded[i, :num_current_encoder_inputs, :] = torch.from_numpy(xyz_centers_np)
            elif num_current_encoder_inputs > 0 : 
                 print(f"WARNUNG: CollateFn - Inkonsistente Anzahl von XYZ-Zentren ({xyz_centers_np.shape[0]}) " \
                       f"zu Features ({num_current_encoder_inputs}) für Sample {i}. Fülle XYZ mit Nullen.")
            
            encoder_input_mask_padded[i, :num_current_encoder_inputs] = False


    # --- 3. Bereite Ground-Truth Targets für den Decoder vor (Padding etc.) ---
    max_gt_objects = 0
    if any(len(sample_gts) > 0 for sample_gts in gt_detections_for_tensor_processing):
         max_gt_objects = max(len(sample_gts) for sample_gts in gt_detections_for_tensor_processing if len(sample_gts) > 0)

    box_dim = 7 
    
    gt_target_boxes_padded = torch.zeros((batch_size, max_gt_objects, box_dim), dtype=torch.float32)
    gt_target_labels_padded = torch.full((batch_size, max_gt_objects), -1, dtype=torch.long) 
    gt_target_valid_mask = torch.zeros((batch_size, max_gt_objects), dtype=torch.bool) 

    for i, sample_gts_list_of_dicts in enumerate(gt_detections_for_tensor_processing):
        if sample_gts_list_of_dicts: 
            current_sample_gt_boxes_7d_list = []
            current_sample_gt_labels_idx_list = []
            for gt_dict in sample_gts_list_of_dicts:
                if 'box_7d' in gt_dict and 'class_idx' in gt_dict:
                    current_sample_gt_boxes_7d_list.append(gt_dict['box_7d'])
                    current_sample_gt_labels_idx_list.append(gt_dict['class_idx'])
                else:
                    print(f"WARNUNG: CollateFn - Fehlende 'box_7d' oder 'class_idx' in gt_dict für Sample {i}. GT-Objekt wird übersprungen.")
            
            if current_sample_gt_boxes_7d_list: # Nur wenn valide GTs vorhanden sind
                current_sample_gt_boxes_7d = torch.tensor(np.array(current_sample_gt_boxes_7d_list), dtype=torch.float32)
                current_sample_gt_labels_idx = torch.tensor(current_sample_gt_labels_idx_list, dtype=torch.long)
                
                num_current_gts = current_sample_gt_boxes_7d.shape[0]
                if num_current_gts > 0:
                    gt_target_boxes_padded[i, :num_current_gts, :] = current_sample_gt_boxes_7d
                    gt_target_labels_padded[i, :num_current_gts] = current_sample_gt_labels_idx
                    gt_target_valid_mask[i, :num_current_gts] = True


    collated_batch = {
        'sample_tokens': sample_tokens,
        'timestamps': torch.tensor(timestamps_np, dtype=torch.float64), # GEÄNDERT: torch.float64
        'ego_translations_world': ego_translations_world,
        'ego_rotations_world_quat': ego_rotations_world_quat,
        
        'encoder_input_features': encoder_input_features_padded,
        'encoder_input_xyz_centers': encoder_input_xyz_centers_padded, 
        'encoder_input_mask': encoder_input_mask_padded,
        
        'gt_target_boxes': gt_target_boxes_padded,
        'gt_target_labels': gt_target_labels_padded,
        'gt_target_valid_mask': gt_target_valid_mask, 
        
        'gt_detections_list_raw': gt_detections_list_raw_batch 
    }
    
    return collated_batch


if __name__ == '__main__':
    print("Starte Test für object_fusion_gt_collate_fn...")
    
    try:
        # Dieser Testblock ist abhängig von der korrekten Funktion von ObjectFusionGTDataset
        # und der Verfügbarkeit der Konfigurationsdatei und des Datensatzes.
        
        # Pfad zur Konfigurationsdatei (relativ zum Projekt-Root /app/)
        config_file_path_collate = "config/pipeline_c_modules.yaml" 
        if not os.path.exists(config_file_path_collate):
            alt_config_path_collate = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'pipeline_c_modules.yaml')
            if os.path.exists(alt_config_path_collate):
                config_file_path_collate = alt_config_path_collate
            else:
                raise FileNotFoundError(f"Config file not found at {config_file_path_collate} or {alt_config_path_collate} for collate_fn test.")

        from oft.utils.config import load_config # Importiere hier, um Abhängigkeit zu reduzieren
        from oft.data.transformer_gt_dataset import ObjectFusionGTDataset # Importiere hier

        full_pipeline_config_dict_collate = load_config(config_file_path_collate)
        print(f"Konfiguration für CollateFn-Test geladen von: {os.path.abspath(config_file_path_collate)}")
        
        dataset_cfg_collate = full_pipeline_config_dict_collate.get('dataset', {})
        dataroot_path_collate = dataset_cfg_collate.get("dataroot", "/app/datasets")
        dataset_version_collate = dataset_cfg_collate.get("version", "v1.0-mini")

        actual_dataset_path_to_check_collate = os.path.join(dataroot_path_collate, dataset_version_collate)
        if not os.path.exists(actual_dataset_path_to_check_collate):
            print(f"FEHLER: Dataset-Pfad '{actual_dataset_path_to_check_collate}' für CollateFn-Test nicht gefunden.")
        else:
            example_dataset = ObjectFusionGTDataset(
                dataroot=dataroot_path_collate,
                version=dataset_version_collate,
                split_name="mini_val", 
                pipeline_config=full_pipeline_config_dict_collate,
                verbose=False 
            )
            
            if len(example_dataset) >= 2:
                print(f"Dataset für CollateFn-Test geladen, {len(example_dataset)} Samples gefunden.")
                num_collate_test_samples = min(4, len(example_dataset))
                dummy_batch_list = [example_dataset[i] for i in range(num_collate_test_samples)]
                
                print(f"\nTeste Collate-Funktion mit Batch der Größe {len(dummy_batch_list)}...")
                collated_batch = object_fusion_gt_collate_fn(dummy_batch_list)
                print("Collate-Funktion erfolgreich ausgeführt.")
                
                print("\nSchlüssel im collated_batch und deren Typen/Shapes:")
                for key, value in collated_batch.items():
                    if torch.is_tensor(value):
                        print(f"  {key}: torch.Tensor, shape {value.shape}, dtype {value.dtype}")
                    elif isinstance(value, list) and value and isinstance(value[0], dict):
                        print(f"  {key}: List of {len(value)} dicts")
                    elif isinstance(value, list):
                        print(f"  {key}: List of {len(value)} items")
                    else:
                        print(f"  {key}: {type(value)}")
                
                assert 'encoder_input_features' in collated_batch and collated_batch['encoder_input_features'].ndim == 3
                assert 'encoder_input_xyz_centers' in collated_batch and collated_batch['encoder_input_xyz_centers'].ndim == 3 and collated_batch['encoder_input_xyz_centers'].shape[-1] == 3
                assert 'timestamps' in collated_batch and collated_batch['timestamps'].dtype == torch.float64
                print("\nWichtige Tensoren (inkl. 'encoder_input_xyz_centers' und korrigiertem 'timestamps'-Typ) sind im Batch vorhanden und haben korrekte Dimensionen.")

            else:
                print(f"Nicht genügend Samples im '{example_dataset.split_name}' Split des Datasets ({len(example_dataset)} gefunden) für einen Batch-Test (benötigt mind. 2).")

    except FileNotFoundError as e:
        print(f"\nError during CollateFn example (Dataset-Initialisierung - FileNotFoundError): {e}")
        print("Stelle sicher, dass der Pfad zur Konfigurationsdatei und zum Datensatz korrekt ist.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during CollateFn example: {e}")
        import traceback
        traceback.print_exc()