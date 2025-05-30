# src/oft/data/transformer_gt_collate.py
import torch
import numpy as np
from typing import List, Dict, Any, Tuple

def object_fusion_gt_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate-Funktion für das ObjectFusionGTDataset.
    Nimmt einen Batch von Samples und bereitet sie für den Transformer-Input vor.
    Fügt 'gt_detections_list_raw' für die Evaluation hinzu.
    """
    
    # --- 1. Sammle alle Datenfelder aus dem Batch ---
    sample_tokens = [item['sample_token'] for item in batch]
    timestamps = [item['timestamp'] for item in batch]
    
    ego_translations_world = torch.tensor(np.array([item['ego_translation_world'] for item in batch]), dtype=torch.float32)
    ego_rotations_world_quat = torch.tensor(np.array([item['ego_rotation_world_quat'] for item in batch]), dtype=torch.float32)

    # Liste der rohen GT-Detektions-Dictionaries pro Sample (für Evaluation benötigt)
    gt_detections_list_raw_batch = [item['gt_detections'] for item in batch]

    # GT-Detektionen für die Verarbeitung zu Tensoren
    gt_detections_for_tensor_processing = [item['gt_detections'] for item in batch]

    history_boxes_batched = [item['history_boxes_7d'] for item in batch]
    history_padding_mask_batched = [item['history_padding_mask'] for item in batch] # Sollte (B,H) sein

    # --- 2. Bestimme maximale Längen für Padding ---
    max_num_gt_detections = 0
    if any(gt_detections_for_tensor_processing):
        max_num_gt_detections = max(len(dets) for dets in gt_detections_for_tensor_processing if dets)
    
    history_len = 0
    if history_boxes_batched and history_boxes_batched[0] is not None and isinstance(history_boxes_batched[0], list): # Prüfe ob es eine Liste ist
        history_len = len(history_boxes_batched[0]) 
    
    max_num_history_boxes_per_frame = 0
    if history_len > 0:
        for sample_history in history_boxes_batched:
            if sample_history: # Stelle sicher, dass sample_history nicht None ist
                for frame_boxes in sample_history:
                    if frame_boxes is not None and isinstance(frame_boxes, np.ndarray): # Prüfe Typ und Existenz
                        max_num_history_boxes_per_frame = max(max_num_history_boxes_per_frame, frame_boxes.shape[0])

    # --- 3. Initialisiere gepaddete Tensoren ---
    num_det_features = 10 
    
    padded_current_features = torch.zeros((len(batch), max_num_gt_detections, num_det_features), dtype=torch.float32)
    padded_current_mask = torch.ones((len(batch), max_num_gt_detections), dtype=torch.bool)
    padded_current_class_labels = torch.full((len(batch), max_num_gt_detections), -1, dtype=torch.long)

    padded_history_boxes = torch.zeros((len(batch), history_len, max_num_history_boxes_per_frame, 7), dtype=torch.float32)
    padded_history_boxes_mask = torch.ones((len(batch), history_len, max_num_history_boxes_per_frame), dtype=torch.bool)

    # --- 4. Fülle die gepaddeten Tensoren ---
    for i, gt_dets_sample in enumerate(gt_detections_for_tensor_processing):
        if gt_dets_sample: 
            num_dets_in_sample = len(gt_dets_sample)
            if num_dets_in_sample > 0:
                sample_boxes = np.array([d['box_world'] for d in gt_dets_sample if 'box_world' in d])
                sample_velocities = np.array([d['velocity_world'] for d in gt_dets_sample if 'velocity_world' in d])
                
                # Stelle sicher, dass sample_boxes und sample_velocities nicht leer sind nach Filterung
                if sample_boxes.size > 0 and sample_velocities.size > 0 and \
                   sample_boxes.shape[0] == sample_velocities.shape[0] and \
                   sample_boxes.ndim == 2 and sample_boxes.shape[1] == 7 and \
                   sample_velocities.ndim == 2 and sample_velocities.shape[1] == 3:
                    
                    sample_features = np.concatenate((sample_boxes, sample_velocities), axis=1)
                    # Stelle sicher, dass wir nicht über die gepaddete Dimension schreiben
                    actual_num_to_copy = min(num_dets_in_sample, max_num_gt_detections)
                    padded_current_features[i, :actual_num_to_copy, :] = torch.from_numpy(sample_features[:actual_num_to_copy])
                    padded_current_mask[i, :actual_num_to_copy] = False 
                
                    sample_labels = np.array([d['class_label'] for d in gt_dets_sample if 'class_label' in d])
                    if sample_labels.size > 0:
                         padded_current_class_labels[i, :actual_num_to_copy] = torch.from_numpy(sample_labels[:actual_num_to_copy])
                # else:
                    # Optional: Warnung, wenn Shapes nicht passen oder Dictionaries unvollständig sind
                    # print(f"Warning: Shape mismatch or missing keys for sample {i} during feature/label creation.")

        sample_hist_frames = history_boxes_batched[i] 
        if sample_hist_frames: # Stelle sicher, dass es nicht None ist
            for frame_idx, hist_frame_boxes_np in enumerate(sample_hist_frames):
                if hist_frame_boxes_np is not None and isinstance(hist_frame_boxes_np, np.ndarray) and hist_frame_boxes_np.shape[0] > 0:
                    num_boxes_in_hist_frame = min(hist_frame_boxes_np.shape[0], max_num_history_boxes_per_frame)
                    padded_history_boxes[i, frame_idx, :num_boxes_in_hist_frame, :] = torch.from_numpy(hist_frame_boxes_np[:num_boxes_in_hist_frame])
                    padded_history_boxes_mask[i, frame_idx, :num_boxes_in_hist_frame] = False
    
    gt_target_boxes = padded_current_features[:, :, :7].clone() 
    gt_target_labels = padded_current_class_labels.clone()
    gt_target_valid_mask = ~padded_current_mask.clone()

    batch_dict = {
        "sample_tokens": sample_tokens,
        "timestamps": timestamps,
        "ego_translations_world": ego_translations_world,
        "ego_rotations_world_quat": ego_rotations_world_quat,
        
        "encoder_input_features": padded_current_features, 
        "encoder_input_mask": padded_current_mask,         
        "encoder_input_class_labels": padded_current_class_labels, 

        "history_boxes_features": padded_history_boxes,       
        "history_boxes_mask": padded_history_boxes_mask,       
        "history_padding_mask_original": torch.tensor(np.array(history_padding_mask_batched, dtype=bool), dtype=torch.bool),

        "gt_target_boxes": gt_target_boxes,             
        "gt_target_labels": gt_target_labels,           
        "gt_target_valid_mask": gt_target_valid_mask,

        # HINZUGEFÜGT für die Evaluation: Liste der rohen GT Dictionaries
        "gt_detections_list_raw": gt_detections_list_raw_batch 
    }

    return batch_dict

if __name__ == '__main__':
    print("Running ObjectFusionGTCollateFn example...")
    
    dummy_pipeline_config_collate = {
        "dataset": {
            "dataroot": "/pfad/zu/deinem/truckscenes", 
            "version": "v1.0-mini",
            "history_window_transformer": 2
        }
    }
    dataroot_path_collate = dummy_pipeline_config_collate["dataset"]["dataroot"]
    dataset_version_collate = dummy_pipeline_config_collate["dataset"]["version"]

    if dataroot_path_collate == "/pfad/zu/deinem/truckscenes":
        print("BITTE ANPASSEN: 'dataroot_path' im Collate-Beispielcode auf deinen lokalen TruckScenes-Pfad setzen, um das Dataset zu laden.")
    else:
        try:
            from oft.data.transformer_gt_dataset import ObjectFusionGTDataset
            
            example_dataset = ObjectFusionGTDataset(
                dataroot=dataroot_path_collate,
                version=dataset_version_collate,
                split_name='mini_train',
                pipeline_config=dummy_pipeline_config_collate,
                verbose=False 
            )

            if len(example_dataset) >= 2:
                sample_batch_list = [example_dataset[0], example_dataset[1]]
                
                print(f"\nCollate-Funktion wird mit einem Batch von {len(sample_batch_list)} Samples getestet.")
                
                collated_batch = object_fusion_gt_collate_fn(sample_batch_list)
                
                print("\nStruktur des collated_batch:")
                for key, value in collated_batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: Tensor shape {value.shape}, dtype {value.dtype}")
                    elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                         print(f"  {key}: List of Tensors, first shape {value[0].shape if value else 'N/A'}")
                    elif isinstance(value, list) and value and isinstance(value[0], dict):
                         print(f"  {key}: List of Dictionaries, {len(value)} items, first item has {len(value[0]) if value[0] else 0} GT detections")
                    elif isinstance(value, list):
                         print(f"  {key}: List with {len(value)} items, first item type: {type(value[0]) if value else 'N/A'}")
                    else:
                        print(f"  {key}: {type(value)}")
                
                assert 'gt_detections_list_raw' in collated_batch
                print("\n'gt_detections_list_raw' ist im Batch vorhanden.")

            else:
                print("Nicht genügend Samples im 'mini_train' Split des Datasets für einen Batch-Test (benötigt mind. 2).")

        except FileNotFoundError as e:
            print(f"\nError during CollateFn example (Dataset-Initialisierung): {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred in CollateFn example: {e}")
            import traceback
            traceback.print_exc()