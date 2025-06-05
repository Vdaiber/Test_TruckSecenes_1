# src/oft/transformer/train.py
"""
Main training script for the ObjectFusionTransformer (OFT) C-Pipeline.
Orchestrates data loading (C1), model (C2, C3), loss (C4), training, and evaluation (C5).
Incorporates set_seed for reproducibility.
Uses updated GT box keys from collate_fn.
Transforms predicted boxes back to world coordinates for DevKit evaluation.
Handles OmegaConf to primitive type conversion for loss/matcher parameters.
Fixes KeyError in SmoothedValue and ValueError for OmegaConf.
Ensures JSON serializability for DevKit outputs.
"""

import argparse
import os
import random
import sys
import time
import json
from pathlib import Path
import datetime
import logging
from typing import Optional, Dict, List, Any
from collections import deque, defaultdict

import numpy as np
import torch
import torch.optim
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig, SCMode

from oft.data.transformer_gt_dataset import ObjectFusionGTDataset
from oft.data.transformer_gt_collate import object_fusion_gt_collate_fn
from oft.transformer.encoder import ObjectEncoder
from oft.transformer.decoder import ObjectFusionTransformerDecoder
from oft.transformer.loss import SetCriterion, HungarianMatcher

from truckscenes import TruckScenes
from truckscenes.eval.detection.evaluate import DetectionEval
from truckscenes.eval.detection.config import DetectionConfig as DevkitDetectionConfig
from truckscenes.eval.detection.data_classes import DetectionBox
from truckscenes.eval.detection.constants import DETECTION_NAMES as TRUCKSCENES_DETECTION_NAMES
from truckscenes.utils.splits import create_splits_scenes # << HINZUGEFÜGTER IMPORT
from pyquaternion import Quaternion as PyQuaternion

# --- Hilfsfunktion zur JSON-Serialisierung ---
def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively converts NumPy data types to native Python types for JSON serialization.
    """
    if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                       np.int16, np.int32, np.int64, np.uint8,
                       np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
        return {'real': float(obj.real), 'imag': float(obj.imag)}
    elif isinstance(obj, (np.ndarray,)):
        return [sanitize_for_json(x) for x in obj.tolist()]
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.void)): # Handles structured arrays if they were to appear
        return None # Or some other appropriate representation
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    return obj


# --- set_seed Funktion ---
def set_seed(seed: Optional[int], logger: Optional[logging.Logger] = None) -> None:
    if seed is not None:
        if logger: logger.info(f"Setting global seed to {seed}")
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if logger: logger.info(f"CUDA seed set for {torch.cuda.device_count()} GPU(s).")
    elif logger:
        logger.info("Seed not set in config, proceeding with random initialization.")

# --- Checkpoint-Funktionen ---
def load_checkpoint(checkpoint_path, model_or_encoder, decoder, optimizer, lr_scheduler, device, logger):
    logger.info(f"Attempting to load checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint file not found: {checkpoint_path}. Starting from scratch.")
        return 0, 0.0
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_loaded_correctly = False
        if isinstance(model_or_encoder, ObjectFusionTransformerModel):
            model_to_load = model_or_encoder.module if hasattr(model_or_encoder, 'module') else model_or_encoder
            if 'encoder_state_dict' in checkpoint and 'decoder_state_dict' in checkpoint:
                 model_to_load.encoder.load_state_dict(checkpoint['encoder_state_dict'])
                 model_to_load.decoder.load_state_dict(checkpoint['decoder_state_dict'])
                 logger.info("Encoder and Decoder states loaded into ObjectFusionTransformerModel.")
                 model_loaded_correctly = True
            elif 'model_state_dict' in checkpoint:
                 logger.info("Attempting to load entire model from 'model_state_dict' (fallback).")
                 model_to_load.load_state_dict(checkpoint['model_state_dict'])
                 logger.info("Entire model state loaded from 'model_state_dict'.")
                 model_loaded_correctly = True
            else:
                 logger.warning("Checkpoint missing 'encoder_state_dict'/'decoder_state_dict' or 'model_state_dict'. Cannot load model.")

        elif isinstance(model_or_encoder, ObjectEncoder) and isinstance(decoder, ObjectFusionTransformerDecoder):
            encoder_to_load = model_or_encoder.module if hasattr(model_or_encoder, 'module') else model_or_encoder
            decoder_to_load = decoder.module if hasattr(decoder, 'module') else decoder
            if 'encoder_state_dict' in checkpoint:
                encoder_to_load.load_state_dict(checkpoint['encoder_state_dict'])
                logger.info("Separate Encoder state loaded.")
                model_loaded_correctly = True
            if 'decoder_state_dict' in checkpoint:
                decoder_to_load.load_state_dict(checkpoint['decoder_state_dict'])
                logger.info("Separate Decoder state loaded.")
                model_loaded_correctly = True
        else:
            logger.error("Invalid model/encoder/decoder combination for load_checkpoint.")
            return 0,0.0

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state loaded.")
        if lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            logger.info("LR Scheduler state loaded.")

        start_epoch = checkpoint.get('epoch', -1) + 1
        best_metric_val = checkpoint.get('best_metric_val', 0.0)
        loaded_config_dict = checkpoint.get('config')
        if loaded_config_dict and logger.isEnabledFor(logging.DEBUG):
             logger.debug("Config from loaded checkpoint:\n" + OmegaConf.to_yaml(OmegaConf.create(loaded_config_dict)))

        logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch}, best previous metric: {best_metric_val:.4f}")
        return start_epoch, best_metric_val
    except Exception as e:
        logger.error(f"Error loading checkpoint from {checkpoint_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.warning("Could not load checkpoint. Starting from scratch.")
        return 0, 0.0

def save_checkpoint(state, is_best, checkpoint_dir, filename_prefix="checkpoint", specific_epoch=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    if specific_epoch is not None:
        filepath = os.path.join(checkpoint_dir, f"{filename_prefix}_epoch_{specific_epoch}.pth.tar")
    else:
        filepath = os.path.join(checkpoint_dir, f"{filename_prefix}_latest.pth.tar")
    torch.save(state, filepath)
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(f"Checkpoint saved to: {filepath}")
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, f"{filename_prefix}_best.pth.tar")
        torch.save(state, best_filepath)
        logging.info(f"Best checkpoint updated and saved to: {best_filepath}")

class ObjectFusionTransformerModel(torch.nn.Module):
    def __init__(self, encoder: ObjectEncoder, decoder: ObjectFusionTransformerDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,
                encoder_input_features: torch.Tensor,
                encoder_input_xyz_centers: torch.Tensor,
                encoder_input_mask: Optional[torch.Tensor] = None
               ) -> Dict[str, torch.Tensor]:
        memory = self.encoder(
            src_features=encoder_input_features,
            src_xyz_centers=encoder_input_xyz_centers,
            src_padding_mask=encoder_input_mask
        )
        predictions = self.decoder(
            memory=memory,
            memory_key_padding_mask=encoder_input_mask
        )
        return predictions

def train_one_epoch(model: ObjectFusionTransformerModel,
                    criterion: SetCriterion,
                    data_loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    cfg_training: Dict[str, Any],
                    logger: logging.Logger):
    model.train()
    criterion.train()

    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}] (Train) '
    print_freq = cfg_training.get("print_freq", 50)

    for batch_idx, batch_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        try:
            encoder_input_features = batch_dict['encoder_input_features'].to(device)
            encoder_input_xyz_centers = batch_dict['encoder_input_xyz_centers'].to(device)
            encoder_input_mask = batch_dict['encoder_input_mask'].to(device)

            gt_target_labels = batch_dict['gt_target_labels'].to(device)
            gt_target_boxes_log_dims = batch_dict['gt_target_boxes_log_dims'].to(device)
            gt_target_boxes_actual_dims = batch_dict['gt_target_boxes_actual_dims'].to(device)
            gt_valid_mask = batch_dict['gt_target_valid_mask'].to(device)

        except KeyError as e:
            logger.error(f"KeyError in train_one_epoch for batch {batch_idx}: {e}. Batch_dict keys: {list(batch_dict.keys())}")
            logger.error("Skipping problematic batch.")
            continue
        except AttributeError as e:
            logger.error(f"AttributeError in train_one_epoch for batch {batch_idx} (likely .to(device) on None): {e}. Batch_dict keys: {list(batch_dict.keys())}")
            logger.error("Skipping problematic batch.")
            continue

        predictions = model(encoder_input_features, encoder_input_xyz_centers, encoder_input_mask)

        losses_dict_unweighted = criterion(predictions,
                                           gt_target_labels,
                                           gt_target_boxes_log_dims,
                                           gt_target_boxes_actual_dims,
                                           gt_valid_mask)

        total_loss = torch.tensor(0.0, device=device)
        if losses_dict_unweighted:
            for loss_name, loss_val_unweighted in losses_dict_unweighted.items():
                weight = criterion.weight_dict.get(loss_name, 1.0)
                total_loss += loss_val_unweighted * weight
                metric_logger.update(**{f'{loss_name}_unscaled': loss_val_unweighted.item()})
                metric_logger.update(**{loss_name: (loss_val_unweighted * weight).item()})
        else:
            logger.warning(f"Epoch {epoch}, Batch {batch_idx}: No losses returned from criterion.")

        metric_logger.update(loss=total_loss.item())

        optimizer.zero_grad()
        if torch.is_tensor(total_loss) and total_loss.requires_grad and not torch.isnan(total_loss) and not torch.isinf(total_loss):
            total_loss.backward()
            if cfg_training.get("clip_max_norm", 0.0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_training.get("clip_max_norm"))
            optimizer.step()
        elif not losses_dict_unweighted:
             logger.warning(f"Epoch {epoch}, Batch {batch_idx}: No loss for backward pass (loss_dict was empty).")
        else:
             logger.error(f"Epoch {epoch}, Batch {batch_idx}: total_loss invalid for backward. Type: {type(total_loss)}, Value: {total_loss}, Requires Grad: {total_loss.requires_grad if torch.is_tensor(total_loss) else 'N/A'}")

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    logger.info(f"Averaged training stats epoch {epoch}: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_model_internally(model: ObjectFusionTransformerModel,
                              criterion: SetCriterion,
                              data_loader: DataLoader,
                              device: torch.device,
                              cfg_dict: Dict[str, Any],
                              logger: logging.Logger,
                              epoch: int):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    header = f'Epoch: [{epoch}] (Validation) '
    print_freq = cfg_dict.get('training', {}).get("print_freq_val", 5)

    all_predictions_for_devkit = []
    all_gt_detections_raw_for_devkit = []
    all_sample_tokens_for_devkit = []

    for batch_idx, batch_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        try:
            encoder_input_features = batch_dict['encoder_input_features'].to(device)
            encoder_input_xyz_centers = batch_dict['encoder_input_xyz_centers'].to(device)
            encoder_input_mask = batch_dict['encoder_input_mask'].to(device)

            gt_target_labels = batch_dict['gt_target_labels'].to(device)
            gt_target_boxes_log_dims = batch_dict['gt_target_boxes_log_dims'].to(device)
            gt_target_boxes_actual_dims = batch_dict['gt_target_boxes_actual_dims'].to(device)
            gt_valid_mask = batch_dict['gt_target_valid_mask'].to(device)

            batch_ego_translations_world = batch_dict['ego_translations_world'].cpu().numpy()
            batch_ego_rotations_world_quat = batch_dict['ego_rotations_world_quat'].cpu().numpy()

        except KeyError as e:
            logger.error(f"KeyError in evaluate_model_internally for batch {batch_idx}: {e}. Batch_dict keys: {list(batch_dict.keys())}")
            logger.error("Skipping problematic validation batch.")
            continue
        except AttributeError as e:
            logger.error(f"AttributeError in evaluate_model_internally for batch {batch_idx}: {e}. Batch_dict keys: {list(batch_dict.keys())}")
            logger.error("Skipping problematic validation batch.")
            continue

        gt_detections_list_raw_batch = batch_dict['gt_detections_list_raw']
        sample_tokens_batch = batch_dict['sample_tokens']

        predictions = model(encoder_input_features, encoder_input_xyz_centers, encoder_input_mask)

        losses_dict_unweighted = criterion(predictions,
                                           gt_target_labels,
                                           gt_target_boxes_log_dims,
                                           gt_target_boxes_actual_dims,
                                           gt_valid_mask)

        total_loss = torch.tensor(0.0, device=device)
        if losses_dict_unweighted:
            loss_weight_dict_eval = cfg_dict.get('loss', {}).get('loss_weight_dict', {})
            for loss_name, loss_val_unweighted in losses_dict_unweighted.items():
                weight = loss_weight_dict_eval.get(loss_name, 1.0)
                total_loss += loss_val_unweighted * weight
                if loss_name in ['loss_ce', 'loss_bbox_l1', 'loss_giou']:
                     metric_logger.update(**{loss_name: (loss_val_unweighted * weight).item()})
        metric_logger.update(loss=total_loss.item())

        pred_logits_batch = predictions['pred_logits']
        pred_boxes_actual_dims_vehicle_batch = predictions['pred_boxes_for_matching_and_giou']

        batch_size_eval = pred_logits_batch.shape[0]
        for i in range(batch_size_eval):
            sample_pred_logits = pred_logits_batch[i]
            sample_pred_boxes_vehicle = pred_boxes_actual_dims_vehicle_batch[i]

            current_sample_token = sample_tokens_batch[i]
            ego_translation_world_np_sample = batch_ego_translations_world[i]
            ego_rotation_world_pyquat_sample = PyQuaternion(batch_ego_rotations_world_quat[i])

            sample_output_boxes_world = []
            scores_all_classes = F.softmax(sample_pred_logits, dim=-1)
            pred_scores, pred_labels_indices = torch.max(scores_all_classes[:, :-1], dim=-1)

            eval_detection_cfg_node = cfg_dict.get('evaluation', {}).get('eval_detection_cfg', {})
            score_threshold = float(eval_detection_cfg_node.get("conf_th_eval", 0.01))

            for q_idx in range(sample_pred_logits.shape[0]):
                if pred_scores[q_idx].item() > score_threshold:
                    center_vehicle = sample_pred_boxes_vehicle[q_idx, 0:3].cpu().numpy()
                    size_vehicle = sample_pred_boxes_vehicle[q_idx, 3:6].cpu().numpy()
                    yaw_vehicle = sample_pred_boxes_vehicle[q_idx, 6].item()

                    center_world = ego_rotation_world_pyquat_sample.rotate(center_vehicle) + ego_translation_world_np_sample
                    ego_yaw_world_sample = ego_rotation_world_pyquat_sample.yaw_pitch_roll[0]
                    yaw_world = yaw_vehicle + ego_yaw_world_sample
                    yaw_world = (yaw_world + np.pi) % (2 * np.pi) - np.pi

                    orientation_world_quat_elements = PyQuaternion(axis=[0, 0, 1], radians=yaw_world).elements

                    label_idx = pred_labels_indices[q_idx].item()
                    class_names_list_resolved = cfg_dict.get('dataset', {}).get('class_names', [])

                    if label_idx < len(class_names_list_resolved):
                        mapped_detection_name = class_names_list_resolved[label_idx]
                    else:
                        mapped_detection_name = "unknown"

                    if mapped_detection_name not in TRUCKSCENES_DETECTION_NAMES:
                        continue

                    det_box = DetectionBox(
                        sample_token=str(current_sample_token),
                        translation=[float(c) for c in center_world],
                        size=[float(s) for s in size_vehicle],
                        rotation=[float(q_el) for q_el in orientation_world_quat_elements],
                        velocity=[0.0, 0.0], # Python floats
                        ego_translation=[float(t) for t in ego_translation_world_np_sample],
                        num_pts=int(-1), # Python int
                        detection_name=str(mapped_detection_name),
                        detection_score=float(pred_scores[q_idx].item()), # Python float
                        attribute_name=""
                    )
                    sample_output_boxes_world.append(det_box.serialize())

            all_predictions_for_devkit.append({"sample_token": current_sample_token, "predictions": sample_output_boxes_world})

            if isinstance(gt_detections_list_raw_batch[i], list):
                all_gt_detections_raw_for_devkit.extend(gt_detections_list_raw_batch[i])
            all_sample_tokens_for_devkit.append(current_sample_token)

    logger.info(f"Averaged validation stats (internal loss) epoch {epoch}: {metric_logger}")

    default_devkit_meta = {"use_lidar": True, "use_camera": False, "use_radar": False, "use_map": False, "use_external": False}
    devkit_meta_node = cfg_dict.get('evaluation', {}).get("devkit_meta", default_devkit_meta)
    meta_for_submission = devkit_meta_node

    results_for_devkit_json = {}
    for item in all_predictions_for_devkit:
        results_for_devkit_json[item['sample_token']] = item['predictions']

    final_submission_dict = {
        "meta": meta_for_submission,
        "results": results_for_devkit_json
    }
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, final_submission_dict, all_gt_detections_raw_for_devkit, all_sample_tokens_for_devkit


def evaluate_model_with_devkit(cfg_dict: Dict[str, Any],
                               prediction_submission_dict: Dict,
                               devkit_dataroot: str,
                               devkit_version: str,
                               devkit_eval_split: str,
                               output_dir_epoch_eval: str,
                               logger: logging.Logger,
                               current_epoch: int):
    eval_output_path_devkit = Path(output_dir_epoch_eval) / "eval_output_devkit"
    eval_output_path_devkit.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    predictions_json_filename = f"predictions_epoch_{current_epoch}_{devkit_eval_split}_{timestamp}.json"
    predictions_json_path = eval_output_path_devkit / predictions_json_filename

    # Sanitize the entire dictionary to ensure all NumPy types are converted
    sanitized_submission_dict = sanitize_for_json(prediction_submission_dict)

    try:
        with open(predictions_json_path, 'w') as f:
            json.dump(sanitized_submission_dict, f, indent=4) # Use sanitized version
    except TypeError as e:
        # This error should ideally not be reached if sanitize_for_json works correctly
        logger.error(f"JSON Serialization Error NACH Sanitize-Versuch: {e}")
        # Log a sample of the problematic data for deeper inspection if necessary
        problematic_sample = None
        if isinstance(sanitized_submission_dict, dict) and "results" in sanitized_submission_dict:
            for sample_tok, preds in sanitized_submission_dict["results"].items():
                if preds: # Log first non-empty prediction list
                    problematic_sample = {sample_tok: preds[0] if isinstance(preds, list) and preds else preds}
                    break
        logger.error(f"Sanitized prediction_submission_dict (Ausschnitt, erstes Sample): {str(problematic_sample)[:1000]}")
        return None

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"evaluate_model_with_devkit: Prediction file: {predictions_json_path}")

    eval_cfg_node = cfg_dict.get('evaluation', {})
    eval_detection_cfg_node = eval_cfg_node.get('eval_detection_cfg', {})

    class_range_from_yaml = eval_detection_cfg_node.get("class_range")
    final_class_range: Dict[str, int]
    if isinstance(class_range_from_yaml, dict):
        final_class_range = class_range_from_yaml
        missing_keys = set(TRUCKSCENES_DETECTION_NAMES) - set(final_class_range.keys())
        if missing_keys:
            for key in missing_keys: final_class_range[key] = 50
    elif class_range_from_yaml is None:
        final_class_range = {name: 50 for name in TRUCKSCENES_DETECTION_NAMES}
    else:
        logger.error(f"DevkitDetectionConfig: 'class_range' unexpected type: {type(class_range_from_yaml)}. Using fallback.")
        final_class_range = {name: 50 for name in TRUCKSCENES_DETECTION_NAMES}

    dist_ths_resolved = eval_detection_cfg_node.get("dist_ths", [0.5, 1.0, 2.0, 4.0])
    default_dist_th_tp = dist_ths_resolved[2] if len(dist_ths_resolved) > 2 else 2.0

    devkit_constructor_params = {
        "class_range": final_class_range,
        "dist_fcn": eval_detection_cfg_node.get("dist_fcn", "center_distance"),
        "dist_ths": dist_ths_resolved,
        "dist_th_tp": eval_detection_cfg_node.get("dist_th_tp", default_dist_th_tp),
        "min_recall": eval_detection_cfg_node.get("min_recall", 0.0),
        "min_precision": eval_detection_cfg_node.get("min_precision", 0.1),
        "max_boxes_per_sample": eval_detection_cfg_node.get("max_boxes_per_sample", 500),
        "mean_ap_weight": eval_detection_cfg_node.get("mean_ap_weight", 5)
    }
    try:
        detection_cfg_for_eval = DevkitDetectionConfig(**devkit_constructor_params)
    except Exception as e:
        logger.error(f"Error creating DevkitDetectionConfig: {e}. Params: {devkit_constructor_params}")
        return None
    try:
        nusc_eval = TruckScenes(version=devkit_version, dataroot=devkit_dataroot, verbose=False)
    except Exception as e:
        logger.error(f"Error initializing TruckScenes for DevKit Eval (dataroot: {devkit_dataroot}, version: {devkit_version}): {e}")
        return None

    if not sanitized_submission_dict or not sanitized_submission_dict.get("results"): # Check sanitized version
        logger.warning(f"Keine Vorhersagen ('results') im (sanitized) prediction_submission_dict für DevKit Eval von Epoche {current_epoch} gefunden. Überspringe DevKit Eval.")
        return None

    eval_set_scenes = create_splits_scenes().get(devkit_eval_split, [])
    eval_set_sample_tokens = []
    if nusc_eval.scene:
        for scene_rec_eval in nusc_eval.scene:
            if scene_rec_eval['name'] in eval_set_scenes:
                s_tok = scene_rec_eval['first_sample_token']
                while s_tok:
                    eval_set_sample_tokens.append(s_tok)
                    s_rec_eval = nusc_eval.get('sample', s_tok)
                    if not s_rec_eval or not s_rec_eval['next']: break
                    s_tok = s_rec_eval['next']

    predicted_sample_tokens = set(sanitized_submission_dict["results"].keys()) # Check sanitized version
    gt_sample_tokens_in_split = set(eval_set_sample_tokens)

    if not predicted_sample_tokens and gt_sample_tokens_in_split :
        logger.warning(f"Vorhersage-Datei für Epoche {current_epoch} enthält keine Samples ('results' ist leer), obwohl GT Samples im Split vorhanden sind. Überspringe DevKit Eval.")
        return None

    if not gt_sample_tokens_in_split:
        logger.warning(f"Keine GT Samples im Eval Split '{devkit_eval_split}' gefunden. Überspringe DevKit Eval.")
        return None

    if predicted_sample_tokens != gt_sample_tokens_in_split and logger.isEnabledFor(logging.WARNING):
        logger.warning(f"Mismatch zwischen vorhergesagten Sample-Tokens ({len(predicted_sample_tokens)}) und GT-Sample-Tokens im Split '{devkit_eval_split}' ({len(gt_sample_tokens_in_split)}).")

    try:
        evaluator = DetectionEval(
            trucksc=nusc_eval, config=detection_cfg_for_eval,
            result_path=str(predictions_json_path), eval_set=devkit_eval_split,
            output_dir=str(eval_output_path_devkit), verbose=eval_detection_cfg_node.get("devkit_verbose",False)
        )
        eval_results = evaluator.main(render_curves=False)
        metrics_summary = eval_results.get('all') if isinstance(eval_results, dict) else None
        return metrics_summary
    except AssertionError as e:
        logger.error(f"AssertionError during DevKit Evaluation for epoch {current_epoch}: {e}")
        logger.error(f"  Pred file: {predictions_json_path}. Check if it's empty or has mismatched/empty sample tokens in 'results'.")
        logger.error(f"  Anzahl vorhergesagter Samples: {len(predicted_sample_tokens)}, Erwartete GT Samples im Split: {len(gt_sample_tokens_in_split)}")
        return None
    except Exception as e:
        logger.error(f"Error during DevKit Evaluation for epoch {current_epoch}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        if fmt is None: fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0; self.count = 0; self.fmt = fmt
    def update(self, value, n=1):
        self.deque.append(value); self.count += n; self.total += value * n
    @property
    def median(self): d = torch.tensor(list(self.deque)); return d.median().item() if len(d) > 0 else 0.0
    @property
    def avg(self): d = torch.tensor(list(self.deque), dtype=torch.float32); return d.mean().item() if len(d) > 0 else 0.0
    @property
    def global_avg(self): return self.total / self.count if self.count > 0 else 0.0
    @property
    def max(self): return max(self.deque) if len(self.deque) > 0 else 0.0
    @property
    def value(self): return self.deque[-1] if len(self.deque) > 0 else 0.0
    def __str__(self):
        if self.count == 0: return "N/A"
        # Stelle sicher, dass alle Format-Schlüssel, die der MetricLogger verwenden könnte, hier verfügbar sind
        return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t", logger=None):
        self.meters = defaultdict(SmoothedValue); self.delimiter = delimiter
        self.logger = logger if logger else logging.getLogger("default_metric_logger")
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor): v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
    def add_meter(self, name: str, meter: SmoothedValue): self.meters[name] = meter
    def __getattr__(self, attr):
        if attr in self.meters: return self.meters[attr]
        if attr in self.__dict__: return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
    def __str__(self): return self.delimiter.join(f"{name}: {str(meter)}" for name, meter in self.meters.items())
    def log_every(self, iterable, print_freq, header=None):
        i = 0; start_time = time.time(); end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}'); data_time = SmoothedValue(fmt='{avg:.4f}')
        try: iterable_len = len(iterable)
        except TypeError: iterable_len = -1
        space_fmt = f':{str(len(str(iterable_len)))}d' if iterable_len > 0 else ''
        log_msg_parts = [header if header else '',
                         '[{0' + space_fmt + '}/{1}]' if iterable_len > 0 else '[{0' + space_fmt + '}]',
                         'eta: {eta}', '{meters}', 'time: {time}', 'data: {data}']
        if torch.cuda.is_available(): log_msg_parts.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg_parts)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            current_len_for_log = iterable_len if iterable_len > 0 else i + 1
            if i % print_freq == 0 or (iterable_len > 0 and i == iterable_len - 1):
                eta_seconds = iter_time.global_avg * (current_len_for_log - 1 - i) if iterable_len > 0 and iter_time.count > 0 else 0
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                format_dict = {"eta": eta_string, "meters": str(self), "time": str(iter_time), "data": str(data_time)}
                if torch.cuda.is_available(): format_dict["memory"] = torch.cuda.max_memory_allocated() / MB
                log_format_args = [i, current_len_for_log] if iterable_len > 0 else [i]
                if self.logger: self.logger.info(log_msg.format(*log_format_args, **format_dict))
            i += 1; end = time.time()
        total_time = time.time() - start_time; total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        avg_time_per_it = total_time / i if i > 0 else 0
        if self.logger: self.logger.info(f'{header if header else ""} Total time: {total_time_str} ({avg_time_per_it:.4f} s / it)')

@hydra.main(config_path="../../../config", config_name="pipeline_c_modules.yaml", version_base=None)
def main(cfg: DictConfig):
    logger = logging.getLogger("train_script_logger")
    log_level_from_cfg = OmegaConf.select(cfg, "training.log_level", default="INFO").upper()
    logger.setLevel(log_level_from_cfg)

    if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if hydra.core.hydra_config.HydraConfig.initialized():
        hydra_logger = logging.getLogger("hydra")
        if hydra_logger:
            hydra_logger.handlers.clear()

    # Wandle OmegaConf zu Python dict direkt am Anfang
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, structured_config_mode=SCMode.DICT_CONFIG)

    if logger.isEnabledFor(logging.INFO): # Logge die aufgelöste Konfiguration nur einmal
        logger.info("Konfiguration (aufgelöst als Python Dict):\n" + json.dumps(cfg_dict, indent=2))


    seed_value = cfg_dict.get('training', {}).get("seed", None)
    set_seed(seed_value, logger)

    device = torch.device(cfg_dict['training']['device'])
    logger.info(f"Verwende Device: {device}")

    dataset_train = ObjectFusionGTDataset(
        dataroot=cfg_dict['dataset']['dataroot'], version=cfg_dict['dataset']['version'],
        split_name=cfg_dict['training']['train_split_name'],
        pipeline_config=cfg_dict,
        verbose=cfg_dict['training'].get("dataset_verbose", True)
    )
    dataset_val = ObjectFusionGTDataset(
        dataroot=cfg_dict['dataset']['dataroot'], version=cfg_dict['dataset']['version'],
        split_name=cfg_dict['training']['val_split_name'],
        pipeline_config=cfg_dict,
        verbose=False
    )

    if cfg_dict['training'].get("debug_sequential_sampler", False) and len(dataset_train) > 0 :
        logger.info("Verwende SequentialSampler für Trainings-DataLoader (Debug-Modus).")
        sampler_train = SequentialSampler(dataset_train)
    elif len(dataset_train) > 0:
         sampler_train = RandomSampler(dataset_train)
    else:
        sampler_train = None

    sampler_val = SequentialSampler(dataset_val) if len(dataset_val) > 0 else None

    dataloader_train = DataLoader(
        dataset_train, batch_size=cfg_dict['training']['batch_size'], sampler=sampler_train,
        collate_fn=object_fusion_gt_collate_fn, num_workers=cfg_dict['training']['num_workers'],
        pin_memory= (device.type == 'cuda')
    ) if sampler_train else None

    dataloader_val = DataLoader(
        dataset_val, batch_size=cfg_dict['training']['batch_size'], sampler=sampler_val,
        collate_fn=object_fusion_gt_collate_fn, num_workers=cfg_dict['training']['num_workers'],
        pin_memory=(device.type == 'cuda')
    ) if sampler_val else None

    logger.info(f"Trainings-Dataset: {len(dataset_train) if dataset_train else 0} Samples, Val-Dataset: {len(dataset_val) if dataset_val else 0} Samples.")
    if not dataloader_train and cfg_dict['training']['epochs'] > 0 :
        logger.error("Trainings-Dataloader konnte nicht erstellt werden. Breche ab.")
        return

    encoder = ObjectEncoder(
        num_input_features=cfg_dict['model']['num_input_features'], d_model=cfg_dict['model']['d_model'],
        nhead=cfg_dict['model']['nhead'], num_encoder_layers=cfg_dict['model']['num_encoder_layers'],
        dim_feedforward=cfg_dict['model']['dim_feedforward_encoder'], dropout=cfg_dict['model']['dropout'],
        activation=cfg_dict['model']['activation'], pe_max_coord_val=cfg_dict['model']['pe_max_coord_val']
    )
    decoder = ObjectFusionTransformerDecoder(
        d_model=cfg_dict['model']['d_model'], nhead=cfg_dict['model']['nhead'],
        num_decoder_layers=cfg_dict['model']['num_decoder_layers'],
        dim_feedforward=cfg_dict['model']['dim_feedforward_decoder'], dropout=cfg_dict['model']['dropout'],
        activation=cfg_dict['model']['activation'], num_queries=cfg_dict['model']['num_queries'],
        num_classes=cfg_dict['model']['num_classes'], box_dim=cfg_dict['model']['box_dim']
    )
    model = ObjectFusionTransformerModel(encoder, decoder).to(device)

    if cfg_dict['training']['use_dataparallel'] and torch.cuda.device_count() > 1:
        logger.info(f"Verwende DataParallel für {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)

    matcher = HungarianMatcher(
        cost_class=float(cfg_dict['loss']['cost_class_weight']),
        cost_bbox_l1=float(cfg_dict['loss']['cost_bbox_l1_weight']),
        cost_giou_bev=float(cfg_dict['loss']['cost_giou_bev_weight'])
    )
    criterion = SetCriterion(
        num_classes=int(cfg_dict['model']['num_classes']),
        matcher=matcher,
        weight_dict=cfg_dict['loss']['loss_weight_dict'],
        eos_coef=float(cfg_dict['loss']['eos_coefficient']),
        losses=cfg_dict['loss']['losses_to_compute'],
        coord_normalization_factor=float(cfg_dict['model']['pe_max_coord_val'])
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg_dict['training']['learning_rate'],
        weight_decay=cfg_dict['training']['weight_decay']
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg_dict['training']['lr_drop_epoch'],
        gamma=cfg_dict['training']['lr_scheduler_gamma']
    )

    start_epoch = 0
    best_metric_val = 0.0

    checkpoint_dir_path_str = cfg_dict['training']['checkpoint_dir']
    if hydra.core.hydra_config.HydraConfig.initialized():
        hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().run.dir
        checkpoint_dir_path = Path(hydra_output_dir) / checkpoint_dir_path_str \
            if not Path(checkpoint_dir_path_str).is_absolute() \
            else Path(checkpoint_dir_path_str)
    else:
        checkpoint_dir_path = Path(checkpoint_dir_path_str)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints werden in '{checkpoint_dir_path}' gespeichert.")


    if cfg_dict['training'].get('resume_checkpoint') and os.path.exists(cfg_dict['training']['resume_checkpoint']):
        start_epoch, best_metric_val = load_checkpoint(cfg_dict['training']['resume_checkpoint'], model, None, optimizer, lr_scheduler, device, logger)
    elif cfg_dict['training'].get('resume_if_checkpoint_exists', False):
        latest_checkpoint_path = checkpoint_dir_path / "oft_c_latest.pth.tar"
        if latest_checkpoint_path.exists():
            logger.info(f"Fortsetzen vom letzten Checkpoint: {latest_checkpoint_path}")
            start_epoch, best_metric_val = load_checkpoint(str(latest_checkpoint_path), model, None, optimizer, lr_scheduler, device, logger)
        else:
            logger.info(f"Kein 'latest' Checkpoint in '{checkpoint_dir_path}' gefunden. Starte von vorne.")
    else:
        logger.info("Starte Training von Epoche 0 (Kein Resume-Checkpoint angegeben oder resume_if_checkpoint_exists=false).")

    logger.info("Starte Trainings-Loop...")
    start_time_total_train = time.time()

    for epoch in range(start_epoch, cfg_dict['training']['epochs']):
        logger.info(f"--- Epoch {epoch}/{cfg_dict['training']['epochs'] -1} ---")
        if dataloader_train:
            train_stats = train_one_epoch(model, criterion, dataloader_train, optimizer, device, epoch, cfg_dict['training'], logger)
        else:
            logger.warning(f"Epoch {epoch}: Trainings-Dataloader ist None, überspringe Trainingsschritt.")
            train_stats = {}

        lr_scheduler.step()

        val_loss_stats = {}
        current_map = float('nan')

        if dataloader_val:
            current_run_dir_val_str = cfg_dict.get('hydra', {}).get('run', {}).get('dir')
            if current_run_dir_val_str:
                 current_run_dir_val = Path(current_run_dir_val_str)
            else:
                 current_run_dir_val = checkpoint_dir_path # Fallback zum Checkpoint-Verzeichnis
            current_epoch_eval_output_dir = current_run_dir_val / f"epoch_{epoch}_eval_outputs"

            val_loss_stats, predictions_for_devkit, _, _ = evaluate_model_internally(
                model, criterion, dataloader_val, device, cfg_dict, logger, epoch
            )

            if predictions_for_devkit and predictions_for_devkit.get("results"):
                devkit_metrics_summary = evaluate_model_with_devkit(
                    cfg_dict=cfg_dict, prediction_submission_dict=predictions_for_devkit,
                    devkit_dataroot=cfg_dict['dataset']['dataroot'], devkit_version=cfg_dict['dataset']['version'],
                    devkit_eval_split=cfg_dict['evaluation']['eval_split_name'],
                    output_dir_epoch_eval=str(current_epoch_eval_output_dir),
                    logger=logger, current_epoch=epoch
                )
                if devkit_metrics_summary and isinstance(devkit_metrics_summary, dict) and 'mean_ap' in devkit_metrics_summary:
                    current_map = devkit_metrics_summary['mean_ap']
                    logger.info(f"Epoch {epoch} - DevKit mAP: {current_map:.4f}")
                else:
                    logger.warning(f"DevKit Eval für Epoche {epoch} hat keine 'mean_ap' geliefert. Summary: {devkit_metrics_summary}")
            else:
                logger.warning(f"Keine Vorhersagen für DevKit-Evaluation in Epoche {epoch} generiert.")
        else:
            logger.warning(f"Epoch {epoch}: Validierungs-Dataloader ist None, überspringe Validierung.")

        is_best = False
        if not np.isnan(current_map) and current_map > best_metric_val :
            best_metric_val = current_map
            is_best = True
            logger.info(f"Neuer bester mAP: {best_metric_val:.4f} in Epoche {epoch}")

        log_stats_epoch = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_loss_stats.items()},
            'epoch': epoch,
            'n_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'val_mAP': current_map if not np.isnan(current_map) else "NaN"
        }

        log_dir_for_stats_str = cfg_dict.get('hydra', {}).get('run', {}).get('dir')
        if log_dir_for_stats_str:
            log_dir_for_stats = Path(log_dir_for_stats_str)
        else:
            log_dir_for_stats = checkpoint_dir_path
        log_dir_for_stats.mkdir(parents=True, exist_ok=True)

        model_to_save = model.module if cfg_dict['training']['use_dataparallel'] and hasattr(model, 'module') else model
        save_dict_content = {
            'encoder_state_dict': model_to_save.encoder.state_dict(),
            'decoder_state_dict': model_to_save.decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'epoch': epoch,
            'best_metric_val': best_metric_val if not np.isnan(best_metric_val) else 0.0,
            'config': cfg_dict
        }
        save_checkpoint(save_dict_content, is_best, str(checkpoint_dir_path), filename_prefix="oft_c")

        if (epoch + 1) % cfg_dict['training']['save_every_k_epochs'] == 0 and epoch > 0:
            save_checkpoint(save_dict_content, False, str(checkpoint_dir_path), filename_prefix="oft_c", specific_epoch=epoch)

        try:
            with (log_dir_for_stats / "log_stats.txt").open("a") as f:
                f.write(json.dumps(log_stats_epoch) + "\n")
        except Exception as e:
            logger.warning(f"Konnte Log-Statistiken nicht nach {log_dir_for_stats / 'log_stats.txt'} schreiben: {e}")

        train_loss_disp = f"{train_stats.get('loss', float('nan')):.4f}" if train_stats and train_stats.get('loss') is not None else "N/A"
        val_loss_disp = f"{val_loss_stats.get('loss', float('nan')):.4f}" if val_loss_stats and val_loss_stats.get('loss') is not None else "N/A"
        map_disp = f"{current_map:.4f}" if not np.isnan(current_map) else "nan"
        logger.info(f"Epoch {epoch} abgeschlossen. Train Loss: {train_loss_disp}, Val Loss: {val_loss_disp}, Val mAP: {map_disp}")

        if cfg_dict['training'].get("run_only_one_epoch_for_debug", False) and epoch == start_epoch :
            logger.warning("DEBUG-Modus: Breche nach einer Epoche ab.")
            break

    total_time_train = time.time() - start_time_total_train
    logger.info(f"Training abgeschlossen. Gesamtzeit: {str(datetime.timedelta(seconds=int(total_time_train)))}")

if __name__ == "__main__":
    main()