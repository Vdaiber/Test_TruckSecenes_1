"""
Main training script for the ObjectFusionTransformer (OFT) C-Pipeline.
Orchestrates data loading (C1), model (C2, C3), loss (C4), training, and evaluation (C5).
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
from collections import deque, defaultdict # Direkter Import

import numpy as np
import torch
import torch.optim
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig

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
from pyquaternion import Quaternion as PyQuaternion

# --- PLATZHALTER FÜR CHECKPOINT-FUNKTIONEN ---
def load_checkpoint(checkpoint_path, model_or_encoder, decoder, optimizer, lr_scheduler, device, logger): # Angepasst für model oder encoder/decoder
    logger.info(f"Attempting to load checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint file not found: {checkpoint_path}. Starting from scratch.")
        return 0, 0.0

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Entscheide, ob das gesamte Modell oder separate Encoder/Decoder geladen werden
        if isinstance(model_or_encoder, ObjectFusionTransformerModel):
            model_or_encoder.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            if decoder is None: # Decoder ist Teil des Modells
                 model_or_encoder.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            else: # Sollte nicht passieren, wenn model_or_encoder ein Gesamtmodell ist
                 decoder.load_state_dict(checkpoint['decoder_state_dict'])

        elif isinstance(model_or_encoder, ObjectEncoder) and isinstance(decoder, ObjectFusionTransformerDecoder):
            model_or_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
        else:
            logger.error("Invalid model/encoder/decoder combination for load_checkpoint.")
            return 0,0.0

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state loaded.")
        if lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            logger.info("LR Scheduler state loaded.")

        start_epoch = checkpoint.get('epoch', -1) + 1 # Starte nächste Epoche
        best_metric_val = checkpoint.get('best_metric_val', 0.0)
        logger.info(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}, best previous metric: {best_metric_val:.4f}")
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
    logging.info(f"Checkpoint saved to: {filepath}")
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, f"{filename_prefix}_best.pth.tar")
        torch.save(state, best_filepath)
        logging.info(f"Best checkpoint updated and saved to: {best_filepath}")
# --- ENDE PLATZHALTER ---


class ObjectFusionTransformerModel(torch.nn.Module):
    """ Wrapper class for the encoder-decoder model. """
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
                    cfg_training: DictConfig, # This is cfg.training
                    logger: logging.Logger):
    model.train()
    criterion.train()

    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}] (Train) '
    print_freq = cfg_training.print_freq

    for batch_idx, batch_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        logger.debug(f"Batch {batch_idx} - Keys: {list(batch_dict.keys())}")
        try:
            encoder_input_features = batch_dict['encoder_input_features'].to(device)
            encoder_input_xyz_centers = batch_dict['encoder_input_xyz_centers'].to(device)
            encoder_input_mask = batch_dict['encoder_input_mask'].to(device)
            gt_target_labels = batch_dict['gt_target_labels'].to(device)
            gt_target_boxes = batch_dict['gt_target_boxes'].to(device)
        except KeyError as e:
            logger.error(f"KeyError in train_one_epoch for batch {batch_idx}: {e}. Batch_dict keys: {list(batch_dict.keys())}")
            logger.error("Skipping problematic batch.")
            continue


        predictions = model(encoder_input_features, encoder_input_xyz_centers, encoder_input_mask)
        losses_dict_unweighted = criterion(predictions, gt_target_labels, gt_target_boxes)

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
        if torch.is_tensor(total_loss) and total_loss.requires_grad:
            total_loss.backward()
            if cfg_training.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_training.clip_max_norm)
            optimizer.step()
        elif not losses_dict_unweighted:
             logger.warning(f"Epoch {epoch}, Batch {batch_idx}: No loss for backward pass (loss_dict was empty).")
        else:
            logger.error(f"Epoch {epoch}, Batch {batch_idx}: total_loss is not a tensor or does not require grad. Type: {type(total_loss)}, Value: {total_loss}")

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    logger.info(f"Averaged training stats epoch {epoch}: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_model_internally(model: ObjectFusionTransformerModel,
                              criterion: SetCriterion,
                              data_loader: DataLoader,
                              device: torch.device,
                              cfg: DictConfig, # This is the full config
                              logger: logging.Logger,
                              epoch: int):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    header = f'Epoch: [{epoch}] (Validation) '
    print_freq = cfg.training.print_freq_val

    all_predictions_for_devkit = []
    all_gt_detections_raw_for_devkit = []
    all_sample_tokens_for_devkit = []

    for batch_idx, batch_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        logger.debug(f"Batch {batch_idx} (Validation) - Keys: {list(batch_dict.keys())}")
        try:
            encoder_input_features = batch_dict['encoder_input_features'].to(device)
            encoder_input_xyz_centers = batch_dict['encoder_input_xyz_centers'].to(device)
            encoder_input_mask = batch_dict['encoder_input_mask'].to(device)
            gt_target_labels = batch_dict['gt_target_labels'].to(device)
            gt_target_boxes = batch_dict['gt_target_boxes'].to(device)
        except KeyError as e:
            logger.error(f"KeyError in evaluate_model_internally for batch {batch_idx}: {e}. Batch_dict keys: {list(batch_dict.keys())}")
            logger.error("Skipping problematic validation batch.")
            continue

        gt_detections_list_raw_batch = batch_dict['gt_detections_list_raw']
        sample_tokens_batch = batch_dict['sample_tokens']

        predictions = model(encoder_input_features, encoder_input_xyz_centers, encoder_input_mask)
        losses_dict_unweighted = criterion(predictions, gt_target_labels, gt_target_boxes)

        total_loss = torch.tensor(0.0, device=device)
        if losses_dict_unweighted:
            for loss_name, loss_val_unweighted in losses_dict_unweighted.items():
                weight = cfg.loss.loss_weight_dict.get(loss_name, 1.0)
                total_loss += loss_val_unweighted * weight
                if loss_name in ['loss_ce', 'loss_bbox_l1', 'loss_giou']:
                     metric_logger.update(**{loss_name: (loss_val_unweighted * weight).item()})
        metric_logger.update(loss=total_loss.item())

        batch_size = predictions['pred_logits'].shape[0]
        for i in range(batch_size):
            sample_pred_logits = predictions['pred_logits'][i]
            sample_pred_boxes = predictions['pred_boxes'][i]

            sample_output_boxes = []

            scores_all_classes = F.softmax(sample_pred_logits, dim=-1)
            pred_scores, pred_labels_indices = torch.max(scores_all_classes[:, :-1], dim=-1)

            score_threshold = cfg.evaluation.eval_detection_cfg.get("conf_th_eval", 0.01)

            for q_idx in range(sample_pred_logits.shape[0]):
                if pred_scores[q_idx] > score_threshold:
                    center = sample_pred_boxes[q_idx, 0:3].cpu().tolist()
                    size = sample_pred_boxes[q_idx, 3:6].cpu().tolist()
                    yaw_rad = sample_pred_boxes[q_idx, 6].item()
                    orientation_quat = PyQuaternion(axis=[0, 0, 1], radians=yaw_rad)

                    label_idx = pred_labels_indices[q_idx].item()
                    class_names_list = OmegaConf.to_container(cfg.dataset.class_names, resolve=True)
                    detailed_detection_name = class_names_list[label_idx] if label_idx < len(class_names_list) else "unknown"

                    mapped_detection_name = detailed_detection_name
                    if detailed_detection_name == "unknown":
                        mapped_detection_name = "unknown"
                    elif detailed_detection_name.startswith("human.pedestrian"):
                        if "pedestrian" in TRUCKSCENES_DETECTION_NAMES: mapped_detection_name = "pedestrian"
                    elif detailed_detection_name == "vehicle.car":
                        if "car" in TRUCKSCENES_DETECTION_NAMES: mapped_detection_name = "car"
                    elif detailed_detection_name == "vehicle.truck" or detailed_detection_name == "vehicle.truck_cabin":
                        if "truck" in TRUCKSCENES_DETECTION_NAMES: mapped_detection_name = "truck"
                    elif detailed_detection_name.startswith("vehicle.bus"):
                        if "bus" in TRUCKSCENES_DETECTION_NAMES: mapped_detection_name = "bus"
                    elif detailed_detection_name == "vehicle.bicycle":
                        if "bicycle" in TRUCKSCENES_DETECTION_NAMES: mapped_detection_name = "bicycle"
                    elif detailed_detection_name == "vehicle.motorcycle":
                        if "motorcycle" in TRUCKSCENES_DETECTION_NAMES: mapped_detection_name = "motorcycle"
                    elif detailed_detection_name == "movable_object.trafficcone":
                        if "traffic_cone" in TRUCKSCENES_DETECTION_NAMES: mapped_detection_name = "traffic_cone"
                        elif "trafficcone" in TRUCKSCENES_DETECTION_NAMES: mapped_detection_name = "trafficcone"
                    elif detailed_detection_name == "movable_object.barrier":
                        if "barrier" in TRUCKSCENES_DETECTION_NAMES: mapped_detection_name = "barrier"
                    elif detailed_detection_name == "vehicle.construction":
                        if "construction_vehicle" in TRUCKSCENES_DETECTION_NAMES: mapped_detection_name = "construction_vehicle"
                    elif detailed_detection_name == "vehicle.trailer":
                        if "trailer" in TRUCKSCENES_DETECTION_NAMES: mapped_detection_name = "trailer"
                    elif detailed_detection_name == "animal": # Added mapping for animal
                        if "animal" in TRUCKSCENES_DETECTION_NAMES: mapped_detection_name = "animal"
                    # Add other mappings for your specific classes if needed, e.g.:
                    # movable_object.debris, movable_object.pushable_pullable,
                    # static_object.bicycle_rack, static_object.bollard, static_object.traffic_island
                    # vehicle.emergency.ambulance, vehicle.emergency.police, vehicle.van, vehicle.wheeled_device
                    # It's crucial that all names in cfg.dataset.class_names are either directly in
                    # TRUCKSCENES_DETECTION_NAMES or are mapped to a name in TRUCKSCENES_DETECTION_NAMES,
                    # or are intentionally mapped to "unknown" if they should be ignored by the devkit.

                    if mapped_detection_name not in TRUCKSCENES_DETECTION_NAMES and mapped_detection_name != "unknown":
                        logger.warning(
                            f"Class name '{mapped_detection_name}' (mapped from '{detailed_detection_name}') "
                            f"is not in TRUCKSCENES_DETECTION_NAMES. Setting to 'unknown'. "
                            f"Available devkit names: {TRUCKSCENES_DETECTION_NAMES}"
                        )
                        mapped_detection_name = "unknown"
                    if detailed_detection_name != "unknown" and mapped_detection_name == "unknown" and detailed_detection_name in TRUCKSCENES_DETECTION_NAMES:
                        logger.info(f"Mapping for '{detailed_detection_name}' resulted in 'unknown', but original is in TRUCKSCENES_DETECTION_NAMES. Using original '{detailed_detection_name}'.")
                        mapped_detection_name = detailed_detection_name

                    det_box = DetectionBox(
                        sample_token=sample_tokens_batch[i],
                        translation=center, size=size, rotation=list(orientation_quat.elements),
                        velocity=[0.0, 0.0], ego_translation=[0.0,0.0,0.0], num_pts=-1,
                        detection_name=mapped_detection_name,
                        detection_score=pred_scores[q_idx].item(),
                        attribute_name=""
                    )
                    sample_output_boxes.append(det_box.serialize())

            all_predictions_for_devkit.append({"sample_token": sample_tokens_batch[i], "predictions": sample_output_boxes})
            if isinstance(gt_detections_list_raw_batch[i], list):
                all_gt_detections_raw_for_devkit.extend(gt_detections_list_raw_batch[i])
            else:
                logger.warning(f"gt_detections_list_raw_batch[{i}] is not a list, but {type(gt_detections_list_raw_batch[i])}. Skipping for DevKit GTs.")
            all_sample_tokens_for_devkit.append(sample_tokens_batch[i])

    logger.info(f"Averaged validation stats (internal loss) epoch {epoch}: {metric_logger}")

    # MODIFIED: Handle devkit_meta correctly
    default_devkit_meta = {"use_lidar": True, "use_camera": False, "use_radar": False, "use_map": False, "use_external": False}
    devkit_meta_node = cfg.evaluation.get("devkit_meta") # Get the node if it exists

    if devkit_meta_node is not None and OmegaConf.is_config(devkit_meta_node):
        # If devkit_meta exists in config and is an OmegaConf node, convert it
        meta_for_submission = OmegaConf.to_container(devkit_meta_node, resolve=True)
    elif devkit_meta_node is not None and isinstance(devkit_meta_node, dict):
        # If devkit_meta exists but was already a dict (e.g. from a non-YAML source or direct modification)
        meta_for_submission = devkit_meta_node
    else:
        # If devkit_meta does not exist in config, use the default Python dict
        meta_for_submission = default_devkit_meta
        logger.info(f"Key 'evaluation.devkit_meta' not found in config. Using default: {meta_for_submission}")


    results_for_devkit_json = {}
    for item in all_predictions_for_devkit:
        results_for_devkit_json[item['sample_token']] = item['predictions']

    final_submission_dict = {
        "meta": meta_for_submission,
        "results": results_for_devkit_json
    }

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, final_submission_dict, all_gt_detections_raw_for_devkit, all_sample_tokens_for_devkit


def evaluate_model_with_devkit(cfg: DictConfig,
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

    logger.info(f"evaluate_model_with_devkit: Prediction file: {predictions_json_path}")

    with open(predictions_json_path, 'w') as f:
        json.dump(prediction_submission_dict, f, indent=4)

    eval_cfg_yaml = cfg.evaluation.eval_detection_cfg

    class_range_from_yaml = eval_cfg_yaml.get("class_range")
    final_class_range: Dict[str, int]

    if isinstance(class_range_from_yaml, (DictConfig, dict)):
        final_class_range = OmegaConf.to_container(class_range_from_yaml, resolve=True)
        missing_keys = set(TRUCKSCENES_DETECTION_NAMES) - set(final_class_range.keys())
        if missing_keys:
            logger.warning(f"DevkitDetectionConfig: 'class_range' from YAML is a dictionary, but is missing keys: {missing_keys}. This will likely cause an error in DetectionConfig. Using default value 50 for missing keys.")
            for key in missing_keys:
                final_class_range[key] = 50
        extra_keys = set(final_class_range.keys()) - set(TRUCKSCENES_DETECTION_NAMES)
        if extra_keys:
            logger.warning(f"DevkitDetectionConfig: 'class_range' from YAML contains extra keys not in TRUCKSCENES_DETECTION_NAMES: {extra_keys}. These will be ignored or might cause errors.")

    elif class_range_from_yaml is None:
        logger.info("DevkitDetectionConfig: 'class_range' is 'null' in YAML. Creating a default class_range with 50m for all TRUCKSCENES_DETECTION_NAMES.")
        final_class_range = {name: 50 for name in TRUCKSCENES_DETECTION_NAMES}
    else:
        logger.error(f"DevkitDetectionConfig: 'class_range' has an unexpected type in YAML: {type(class_range_from_yaml)}. Expected Dict or Null. Using fallback default.")
        final_class_range = {name: 50 for name in TRUCKSCENES_DETECTION_NAMES}

    dist_ths_resolved = OmegaConf.to_container(eval_cfg_yaml.dist_ths, resolve=True) if isinstance(eval_cfg_yaml.dist_ths, (ListConfig, list)) else [0.5, 1.0, 2.0, 4.0]

    default_dist_th_tp = dist_ths_resolved[2] if len(dist_ths_resolved) > 2 else 2.0
    default_min_precision = 0.1
    default_mean_ap_weight = 5

    dist_th_tp_resolved = eval_cfg_yaml.get("dist_th_tp", default_dist_th_tp)
    if not eval_cfg_yaml.get("dist_th_tp"):
        logger.warning(f"DevkitDetectionConfig: 'dist_th_tp' not found in YAML. Using default value: {default_dist_th_tp}")

    min_precision_resolved = eval_cfg_yaml.get("min_precision", default_min_precision)
    if not eval_cfg_yaml.get("min_precision"):
        logger.warning(f"DevkitDetectionConfig: 'min_precision' not found in YAML. Using default value: {default_min_precision}")

    mean_ap_weight_resolved = eval_cfg_yaml.get("mean_ap_weight", default_mean_ap_weight)
    if not eval_cfg_yaml.get("mean_ap_weight"):
         logger.warning(f"DevkitDetectionConfig: 'mean_ap_weight' not found in YAML. Using default value: {default_mean_ap_weight}")

    devkit_constructor_params = {
        "class_range": final_class_range,
        "dist_fcn": eval_cfg_yaml.get("dist_fcn", "center_distance"),
        "dist_ths": dist_ths_resolved,
        "dist_th_tp": dist_th_tp_resolved,
        "min_recall": eval_cfg_yaml.get("min_recall", 0.0),
        "min_precision": min_precision_resolved,
        "max_boxes_per_sample": eval_cfg_yaml.get("max_boxes_per_sample", 500),
        "mean_ap_weight": mean_ap_weight_resolved
    }
    for param_name_check in ["dist_fcn", "min_recall", "max_boxes_per_sample"]:
        if not eval_cfg_yaml.get(param_name_check):
             logger.warning(f"DevkitDetectionConfig: Parameter '{param_name_check}' for DetectionConfig constructor was not found in YAML. Using default value: {devkit_constructor_params[param_name_check]}")

    try:
        detection_cfg_for_eval = DevkitDetectionConfig(**devkit_constructor_params)
        logger.info(f"DevkitDetectionConfig successfully created for epoch {current_epoch}.")
    except Exception as e:
        logger.error(f"Error creating DevkitDetectionConfig for epoch {current_epoch}: {e}")
        logger.error(f"Used constructor parameters: {devkit_constructor_params}")
        import traceback
        logger.error(traceback.format_exc())
        return None

    try:
        nusc_eval = TruckScenes(version=devkit_version, dataroot=devkit_dataroot, verbose=False)
    except Exception as e:
        logger.error(f"Error initializing TruckScenes for DevKit Eval (dataroot: {devkit_dataroot}, version: {devkit_version}): {e}")
        return None

    try:
        evaluator = DetectionEval(
            trucksc=nusc_eval,
            config=detection_cfg_for_eval,
            result_path=str(predictions_json_path),
            eval_set=devkit_eval_split,
            output_dir=str(eval_output_path_devkit),
            verbose=True
        )

        eval_results = evaluator.main(render_curves=False)

        metrics_summary = None
        if isinstance(eval_results, dict) and 'all' in eval_results and isinstance(eval_results['all'], dict):
            metrics_summary = eval_results['all']
        elif isinstance(eval_results, dict):
             metrics_summary = eval_results
        else:
            logger.warning(f"Unexpected return type or structure from DetectionEval.main(): {type(eval_results)}. Metrics might not be extracted correctly.")

        logger.info(f"DevKit Evaluation for epoch {current_epoch} completed.")
        if metrics_summary:
            logger.info(f"Metrics Summary (epoch {current_epoch}): {metrics_summary}")
        return metrics_summary
    except Exception as e:
        logger.error(f"Error during DevKit Evaluation for epoch {current_epoch}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item() if len(self.deque) > 0 else 0.0

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item() if len(self.deque) > 0 else 0.0

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def max(self):
        return max(self.deque) if len(self.deque) > 0 else 0.0

    @property
    def value(self):
        return self.deque[-1] if len(self.deque) > 0 else 0.0

    def __str__(self):
        if self.count == 0: return "N/A"
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg,
            max=self.max, value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t", logger=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logger = logger if logger else logging.getLogger("default_metric_logger")

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def add_meter(self, name: str, meter: SmoothedValue):
        self.meters[name] = meter

    def __getattr__(self, attr):
        if attr in self.meters: return self.meters[attr]
        if attr in self.__dict__: return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        return self.delimiter.join(f"{name}: {str(meter)}" for name, meter in self.meters.items())

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header: header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')

        try: iterable_len = len(iterable)
        except TypeError: iterable_len = -1

        space_fmt = f':{str(len(str(iterable_len)))}d' if iterable_len > 0 else ''
        log_msg_parts = [header]
        log_msg_parts.append('[{0' + space_fmt + '}/{1}]' if iterable_len > 0 else '[{0' + space_fmt + '}]')
        log_msg_parts.extend(['eta: {eta}', '{meters}', 'time: {time}', 'data: {data}'])
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

                if iterable_len > 0:
                    self.logger.info(log_msg.format(i, current_len_for_log, **format_dict))
                else:
                    self.logger.info(log_msg.format(i, **format_dict))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        avg_time_per_it = total_time / i if i > 0 else 0
        self.logger.info(f'{header} Total time: {total_time_str} ({avg_time_per_it:.4f} s / it)')


@hydra.main(config_path="../../../config", config_name="pipeline_c_modules.yaml", version_base=None)
def main(cfg: DictConfig):
    logger = logging.getLogger("train_script_logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.info("Konfiguration:\n" + OmegaConf.to_yaml(cfg))

    seed = cfg.training.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(cfg.training.device)
    logger.info(f"Verwende Device: {device}")

    dataset_train = ObjectFusionGTDataset(
        dataroot=cfg.dataset.dataroot, version=cfg.dataset.version,
        split_name=cfg.training.train_split_name,
        pipeline_config=OmegaConf.to_container(cfg, resolve=True), verbose=True
    )
    dataset_val = ObjectFusionGTDataset(
        dataroot=cfg.dataset.dataroot, version=cfg.dataset.version,
        split_name=cfg.training.val_split_name,
        pipeline_config=OmegaConf.to_container(cfg, resolve=True), verbose=False
    )

    sampler_train = RandomSampler(dataset_train) if len(dataset_train) > 0 else None
    sampler_val = SequentialSampler(dataset_val) if len(dataset_val) > 0 else None

    dataloader_train = DataLoader(
        dataset_train, batch_size=cfg.training.batch_size, sampler=sampler_train,
        collate_fn=object_fusion_gt_collate_fn, num_workers=cfg.training.num_workers,
        pin_memory=device.type == 'cuda'
    ) if sampler_train else None

    dataloader_val = DataLoader(
        dataset_val, batch_size=cfg.training.batch_size, sampler=sampler_val,
        collate_fn=object_fusion_gt_collate_fn, num_workers=cfg.training.num_workers,
        pin_memory=device.type == 'cuda'
    ) if sampler_val else None

    logger.info(f"Trainings-Dataset: {len(dataset_train)} Samples, Val-Dataset: {len(dataset_val)} Samples.")
    if not dataloader_train and cfg.training.epochs > 0 :
        logger.error("Trainings-Dataloader konnte nicht erstellt werden (Dataset leer), aber Training ist angefordert. Breche ab.")
        return

    encoder = ObjectEncoder(
        num_input_features=cfg.model.num_input_features, d_model=cfg.model.d_model,
        nhead=cfg.model.nhead, num_encoder_layers=cfg.model.num_encoder_layers,
        dim_feedforward=cfg.model.dim_feedforward_encoder, dropout=cfg.model.dropout,
        activation=cfg.model.activation, pe_max_coord_val=cfg.model.pe_max_coord_val
    )
    decoder = ObjectFusionTransformerDecoder(
        d_model=cfg.model.d_model, nhead=cfg.model.nhead,
        num_decoder_layers=cfg.model.num_decoder_layers,
        dim_feedforward=cfg.model.dim_feedforward_decoder, dropout=cfg.model.dropout,
        activation=cfg.model.activation, num_queries=cfg.model.num_queries,
        num_classes=cfg.model.num_classes, box_dim=cfg.model.box_dim
    )
    model = ObjectFusionTransformerModel(encoder, decoder).to(device)

    if cfg.training.use_dataparallel and torch.cuda.device_count() > 1:
        logger.info(f"Verwende DataParallel für {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)

    matcher = HungarianMatcher(
        cost_class=cfg.loss.cost_class_weight, cost_bbox_l1=cfg.loss.cost_bbox_l1_weight,
        cost_giou_bev=cfg.loss.cost_giou_bev_weight
    )
    criterion = SetCriterion(
        num_classes=cfg.model.num_classes,
        matcher=matcher,
        weight_dict=OmegaConf.to_container(cfg.loss.loss_weight_dict, resolve=True),
        eos_coef=cfg.loss.eos_coefficient,
        losses=OmegaConf.to_container(cfg.loss.losses_to_compute, resolve=True)
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.training.lr_drop_epoch,
        gamma=cfg.training.lr_scheduler_gamma
    )

    start_epoch = 0
    best_metric_val = 0.0
    checkpoint_dir_path = Path(cfg.training.checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    if cfg.training.resume_checkpoint:
        if os.path.exists(cfg.training.resume_checkpoint):
            model_to_load = model.module if cfg.training.use_dataparallel and hasattr(model, 'module') else model
            start_epoch, best_metric_val = load_checkpoint(cfg.training.resume_checkpoint, model_to_load, None, optimizer, lr_scheduler, device, logger)
        else:
            logger.warning(f"Resume-Checkpoint {cfg.training.resume_checkpoint} nicht gefunden. Starte von vorne.")
    elif cfg.training.resume_if_checkpoint_exists:
        latest_checkpoint_path = checkpoint_dir_path / "oft_c_checkpoint_latest.pth.tar"
        if latest_checkpoint_path.exists():
            logger.info(f"Fortsetzen vom letzten Checkpoint: {latest_checkpoint_path}")
            model_to_load = model.module if cfg.training.use_dataparallel and hasattr(model, 'module') else model
            start_epoch, best_metric_val = load_checkpoint(str(latest_checkpoint_path), model_to_load, None, optimizer, lr_scheduler, device, logger)
        else:
            logger.info("Kein 'latest' Checkpoint gefunden. Starte von vorne.")

    logger.info("Starte Trainings-Loop...")
    start_time = time.time()

    for epoch in range(start_epoch, cfg.training.epochs):
        logger.info(f"--- Epoch {epoch}/{cfg.training.epochs -1} ---")

        if dataloader_train:
            train_stats = train_one_epoch(model, criterion, dataloader_train, optimizer, device, epoch, cfg.training, logger)
        else:
            logger.warning(f"Epoch {epoch}: Trainings-Dataloader ist None, überspringe Trainingsschritt.")
            train_stats = {}

        lr_scheduler.step()

        val_loss_stats = {}
        current_map = float('nan')

        if dataloader_val:
            val_loss_stats, predictions_for_devkit, _, _ = evaluate_model_internally(
                model, criterion, dataloader_val, device, cfg, logger, epoch
            )
            current_epoch_eval_output_dir = checkpoint_dir_path / f"epoch_{epoch}_eval_outputs"
            current_epoch_eval_output_dir.mkdir(parents=True, exist_ok=True)

            devkit_metrics_summary = evaluate_model_with_devkit(
                cfg=cfg, prediction_submission_dict=predictions_for_devkit,
                devkit_dataroot=cfg.dataset.dataroot, devkit_version=cfg.dataset.version,
                devkit_eval_split=cfg.evaluation.eval_split_name,
                output_dir_epoch_eval=str(current_epoch_eval_output_dir),
                logger=logger, current_epoch=epoch
            )
            if devkit_metrics_summary and isinstance(devkit_metrics_summary, dict) and 'mean_ap' in devkit_metrics_summary:
                current_map = devkit_metrics_summary['mean_ap']
                logger.info(f"Epoch {epoch} - DevKit mAP: {current_map:.4f}")
            else:
                logger.warning(f"DevKit Evaluation für Epoche {epoch} hat keine 'mean_ap'-Metrik im Summary geliefert oder devkit_metrics_summary ist None/unerwartet. Summary: {devkit_metrics_summary}")
        else:
            logger.warning(f"Epoch {epoch}: Validierungs-Dataloader ist None, überspringe Validierungs- und Evaluationsschritt.")

        if not np.isnan(current_map) and current_map > best_metric_val :
            best_metric_val = current_map
            is_best = True
            logger.info(f"Neuer bester mAP: {best_metric_val:.4f} in Epoche {epoch}")
        else:
            is_best = False

        log_stats_epoch = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_loss_stats.items()},
            'epoch': epoch,
            'n_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'val_mAP': current_map if not np.isnan(current_map) else "NaN"
        }
        log_dir_str = cfg.get("hydra", {}).get("runtime", {}).get("output_dir", str(checkpoint_dir_path))
        log_dir = Path(log_dir_str)
        log_dir.mkdir(parents=True, exist_ok=True)

        model_to_save = model.module if cfg.training.use_dataparallel and hasattr(model, 'module') else model
        save_dict_content = {
            'encoder_state_dict': model_to_save.encoder.state_dict(),
            'decoder_state_dict': model_to_save.decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'epoch': epoch,
            'best_metric_val': best_metric_val if not np.isnan(best_metric_val) else 0.0,
            'config': OmegaConf.to_container(cfg, resolve=True)
        }
        save_checkpoint(save_dict_content, is_best, checkpoint_dir_path, filename_prefix="oft_c")

        if (epoch + 1) % cfg.training.save_every_k_epochs == 0 and epoch > 0:
            save_checkpoint(save_dict_content, False, checkpoint_dir_path, filename_prefix="oft_c", specific_epoch=epoch)

        try:
            with (log_dir / "log_stats.txt").open("a") as f:
                f.write(json.dumps(log_stats_epoch) + "\n")
        except Exception as e:
            logger.warning(f"Konnte Log-Statistiken nicht nach {log_dir / 'log_stats.txt'} schreiben: {e}")

        train_loss_display_str = f"{train_stats.get('loss', float('nan')):.4f}" if train_stats and train_stats.get('loss') is not None else "N/A"
        val_loss_display_str = f"{val_loss_stats.get('loss', float('nan')):.4f}" if val_loss_stats and val_loss_stats.get('loss') is not None else "N/A"
        current_map_display_str = f"{current_map:.4f}" if not np.isnan(current_map) else "nan"
        logger.info(f"Epoch {epoch} abgeschlossen. Train Loss: {train_loss_display_str}, Val Loss: {val_loss_display_str}, Val mAP: {current_map_display_str}")

        if cfg.training.get("run_only_one_epoch_for_debug", False) and epoch == start_epoch :
            logger.warning("DEBUG-Modus: Breche nach einer Epoche ab.")
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training abgeschlossen. Gesamtzeit: {total_time_str}")

if __name__ == "__main__":
    main()