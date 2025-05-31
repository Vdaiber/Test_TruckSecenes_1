# src/oft/transformer/train.py
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import argparse 
from typing import Dict, Any, Optional, Tuple, List
import numpy as np 
import torch.nn.functional as F 
import json 

# Korrekte relative Importe basierend auf der Projektstruktur
from oft.utils.config import load_config
from oft.data.transformer_gt_dataset import ObjectFusionGTDataset
from oft.data.transformer_gt_collate import object_fusion_gt_collate_fn
from oft.transformer.encoder import ObjectEncoder
from oft.transformer.decoder import ObjectFusionTransformerDecoder
from oft.transformer.loss import HungarianMatcher, SetCriterion

# Importe für die TruckScenes Evaluation
from truckscenes import TruckScenes 
from truckscenes.eval.detection.config import config_factory 
from truckscenes.eval.detection.data_classes import DetectionConfig 
from truckscenes.eval.detection.evaluate import DetectionEval 
from truckscenes.eval.common.data_classes import EvalBoxes
from pyquaternion import Quaternion as PyQuaternion

class ObjectFusionTransformerModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, 
                src_features: torch.Tensor, 
                src_padding_mask: Optional[torch.Tensor], 
                src_xyz_centers: torch.Tensor,
               ) -> Dict[str, torch.Tensor]:
        memory = self.encoder(
            src_features=src_features, 
            src_padding_mask=src_padding_mask, 
            src_xyz_centers=src_xyz_centers
        )
        outputs = self.decoder(
            memory=memory, 
            memory_key_padding_mask=src_padding_mask
        )
        return outputs

def save_checkpoint(epoch: int, model: nn.Module, optimizer: optim.Optimizer, loss: float, filepath: str, config: Optional[Dict] = None):
    print(f"=> Saving checkpoint for epoch {epoch+1} to {filepath}")
    model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    state = {'epoch': epoch, 'state_dict': model_state_dict, 'optimizer': optimizer.state_dict(), 'loss': loss, 'config': config}
    torch.save(state, filepath)

def load_checkpoint(filepath: str, model: nn.Module, optimizer: Optional[optim.Optimizer] = None, device: Optional[torch.device] = None) -> Tuple[int, float, Optional[Dict]]:
    if not os.path.isfile(filepath):
        print(f"=> No checkpoint found at '{filepath}'")
        return 0, float('inf'), None
    print(f"=> Loading checkpoint '{filepath}'")
    checkpoint = torch.load(filepath, map_location=device if device else 'cpu')
    start_epoch = checkpoint.get('epoch', -1) + 1
    state_dict = checkpoint['state_dict']
    current_is_dp = isinstance(model, nn.DataParallel)
    saved_is_dp = any(k.startswith('module.') for k in state_dict.keys())
    if current_is_dp and not saved_is_dp:
        new_state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        state_dict = new_state_dict
    elif not current_is_dp and saved_is_dp:
        new_state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith('module.')}
        for k, v in state_dict.items():
            if not k.startswith('module.'): new_state_dict[k] = v
        state_dict = new_state_dict
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Warning: Error loading state_dict strictly: {e}. Trying non-strict.")
        model.load_state_dict(state_dict, strict=False)
    if optimizer and checkpoint.get('optimizer'):
        try: optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError as e: print(f"Warning: Could not load optimizer state: {e}")
    loss = checkpoint.get('loss', float('inf'))
    loaded_config = checkpoint.get('config')
    print(f"=> Loaded checkpoint '{filepath}' (resuming from epoch {start_epoch}, last val_loss {loss:.4f})")
    return start_epoch, loss, loaded_config

def train_one_epoch(model: nn.Module, criterion: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, device: torch.device, epoch: int, cfg: Dict[str, Any], print_freq: int = 10):
    model.train(); criterion.train()
    running_loss, total_loss_sum, num_batches = 0.0, 0.0, len(dataloader)
    epoch_start_time, batch_start_time = time.time(), time.time()
    for batch_idx, batch_data in enumerate(dataloader):
        encoder_input_features = batch_data["encoder_input_features"].to(device, non_blocking=True)
        encoder_input_mask = batch_data["encoder_input_mask"].to(device, non_blocking=True)
        src_xyz_centers = encoder_input_features[:, :, :3].clone()
        gt_labels = batch_data["gt_target_labels"].to(device, non_blocking=True)
        gt_boxes = batch_data["gt_target_boxes"].to(device, non_blocking=True)
        gt_valid_mask = batch_data["gt_target_valid_mask"].to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs_dict = model(src_features=encoder_input_features, src_padding_mask=encoder_input_mask, src_xyz_centers=src_xyz_centers)
        loss_dict = criterion(outputs_dict, gt_valid_mask, gt_labels, gt_boxes)
        weight_dict = criterion.weight_dict
        total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        total_loss.backward()
        max_grad_norm = cfg.get("training", {}).get("clip_max_norm", None)
        if max_grad_norm is not None and max_grad_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        current_batch_loss = total_loss.item()
        running_loss += current_batch_loss; total_loss_sum += current_batch_loss
        if (batch_idx + 1) % print_freq == 0 or batch_idx == num_batches - 1:
            current_lr = optimizer.param_groups[0]['lr']
            avg_batch_time = (time.time() - batch_start_time) / min(print_freq, batch_idx + 1)
            print(f"Epoch: [{epoch+1}][{batch_idx+1}/{num_batches}]\tLoss: {current_batch_loss:.4f} (Avg last {min(print_freq, batch_idx + 1)}: {running_loss/min(print_freq, batch_idx+1):.4f})\tLR: {current_lr:.1e}\tTime/Batch: {avg_batch_time:.3f}s")
            running_loss = 0.0; batch_start_time = time.time()
            print(f"  Losses: " + " | ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()]))
    avg_epoch_loss = total_loss_sum / num_batches
    epoch_time = time.time() - epoch_start_time
    print(f"--- Epoch {epoch+1} Training Summary ---\nAverage Training Loss: {avg_epoch_loss:.4f}\nTime taken: {epoch_time:.2f}s ({epoch_time/num_batches:.3f}s/batch)")
    return avg_epoch_loss

@torch.no_grad()
def validate_one_epoch(model: nn.Module, criterion: nn.Module, dataloader: DataLoader, device: torch.device, epoch: int, cfg: Dict[str, Any]):
    model.eval(); criterion.eval()
    total_loss_sum = 0.0
    num_batches = len(dataloader) 
    display_epoch = epoch + 1 if epoch >=0 else "Eval" 
    print(f"\n--- Starting Validation for Epoch {display_epoch} ---")

    start_time = time.time()
    for batch_idx, batch_data in enumerate(dataloader):
        encoder_input_features = batch_data["encoder_input_features"].to(device, non_blocking=True)
        encoder_input_mask = batch_data["encoder_input_mask"].to(device, non_blocking=True)
        src_xyz_centers = encoder_input_features[:, :, :3].clone()
        gt_labels = batch_data["gt_target_labels"].to(device, non_blocking=True)
        gt_boxes = batch_data["gt_target_boxes"].to(device, non_blocking=True)
        gt_valid_mask = batch_data["gt_target_valid_mask"].to(device, non_blocking=True)
        outputs_dict = model(src_features=encoder_input_features, src_padding_mask=encoder_input_mask, src_xyz_centers=src_xyz_centers)
        loss_dict = criterion(outputs_dict, gt_valid_mask, gt_labels, gt_boxes)
        weight_dict = criterion.weight_dict
        total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        total_loss_sum += total_loss.item()
        log_freq_val = cfg.get("training", {}).get("print_freq_val", max(1, num_batches // 2 if num_batches > 0 else 1))
        if (batch_idx + 1) % log_freq_val == 0 or batch_idx == num_batches - 1 : print(f"Validation: [{batch_idx+1}/{num_batches}]\tLoss: {total_loss.item():.4f}")
    
    avg_val_loss = total_loss_sum / num_batches if num_batches > 0 else 0.0
    val_time = time.time() - start_time
    avg_batch_time_val = val_time / num_batches if num_batches > 0 else 0.0
    print(f"--- Epoch {display_epoch} Validation Summary ---\nAverage Validation Loss: {avg_val_loss:.4f}\nTime taken: {val_time:.2f}s ({avg_batch_time_val:.3f}s/batch)\n")
    return avg_val_loss

@torch.no_grad()
def evaluate_model(model: nn.Module, 
                   dataloader: DataLoader, 
                   device: torch.device, 
                   cfg: Dict[str, Any],
                   # ts_instance: TruckScenes, # Entfernt, da für diese DetectionEval-API nicht direkt benötigt
                   criterion: Optional[nn.Module] = None): 
    print("\n--- Starting Evaluation with TruckScenes DevKit ---") 
    model.eval()

    eval_main_cfg = cfg.get("evaluation", {})
    eval_params_from_yaml = eval_main_cfg.get("eval_detection_cfg", {})
    dataset_cfg = cfg.get("dataset", {})
    
    class_names = dataset_cfg.get("class_names")
    if not class_names or not isinstance(class_names, list) or len(class_names) == 0:
        print("FEHLER: 'dataset.class_names' nicht in der Konfiguration gefunden oder ungültig. Evaluation nicht möglich.")
        return {"mAP": -1.0, "error": "class_names missing or invalid in config"}

    devkit_base_config_name = eval_params_from_yaml.get("devkit_base_config_name", "detection_cvpr_2024")
    try:
        detection_config_obj = config_factory(devkit_base_config_name)
        print(f"Basis-Evaluationskonfiguration '{devkit_base_config_name}' aus DevKit geladen.")
    except Exception as e:
        print(f"WARNUNG: Konnte DevKit-Basis-Eval-Config '{devkit_base_config_name}' nicht laden: {e}.")
        detection_config_obj = DetectionConfig()

    # Überschreibe/Setze Parameter aus unserer pipeline_c_modules.yaml
    detection_config_obj.class_names = class_names 
    if "dist_fcn" in eval_params_from_yaml: detection_config_obj.dist_fcn = eval_params_from_yaml["dist_fcn"]
    if "dist_ths" in eval_params_from_yaml: detection_config_obj.dist_ths = np.array(eval_params_from_yaml["dist_ths"])
    if "iou_ths_bev" in eval_params_from_yaml: detection_config_obj.iou_ths = np.array(eval_params_from_yaml["iou_ths_bev"])
    if "conf_th_eval" in eval_params_from_yaml: detection_config_obj.conf_th = eval_params_from_yaml["conf_th_eval"]
    if "min_recall" in eval_params_from_yaml: detection_config_obj.min_recall = eval_params_from_yaml["min_recall"]
    if "max_boxes_per_sample" in eval_params_from_yaml: detection_config_obj.max_boxes_per_sample = eval_params_from_yaml["max_boxes_per_sample"]
    if "min_points_per_box" in eval_params_from_yaml: detection_config_obj.min_points = eval_params_from_yaml["min_points_per_box"]
    if "step_size_recall_pr" in eval_params_from_yaml: detection_config_obj.step_size_recall_pr = eval_params_from_yaml["step_size_recall_pr"]
    
    all_pred_eval_boxes = EvalBoxes() 
    all_gt_eval_boxes = EvalBoxes() 
    num_model_classes = cfg.get("model", {}).get("num_classes", len(class_names))
    
    num_batches = len(dataloader)
    if num_batches == 0:
        print("FEHLER: Eval Dataloader ist leer.")
        return {"mAP": -1.0, "error": "Eval dataloader empty"}

    print("Sammle Modellvorhersagen und GT-Daten für Evaluation...")
    for batch_idx, batch_data in enumerate(dataloader):
        encoder_input_features = batch_data["encoder_input_features"].to(device, non_blocking=True)
        encoder_input_mask = batch_data["encoder_input_mask"].to(device, non_blocking=True)
        src_xyz_centers = encoder_input_features[:, :, :3].clone()
        
        if 'gt_detections_list_raw' not in batch_data:
            print("FEHLER: 'gt_detections_list_raw' nicht im Batch gefunden. Bitte Collate-Funktion anpassen.")
            return {"mAP": -1.0, "error": "gt_detections_list_raw missing"}
            
        raw_gt_detections_batch = batch_data['gt_detections_list_raw'] 
        batch_sample_tokens = batch_data["sample_tokens"]
        
        outputs_dict = model(
            src_features=encoder_input_features,
            src_padding_mask=encoder_input_mask,
            src_xyz_centers=src_xyz_centers
        )
        
        pred_logits_batch = outputs_dict['pred_logits'].cpu() 
        pred_boxes_batch = outputs_dict['pred_boxes'].cpu()   

        for i in range(len(batch_sample_tokens)): 
            sample_token = batch_sample_tokens[i]
            pred_logits_sample = pred_logits_batch[i] 
            pred_boxes_sample = pred_boxes_batch[i]   

            sample_pred_boxes_for_eval_list = []
            pred_scores_softmax = F.softmax(pred_logits_sample, dim=-1) 
            
            for q_idx in range(pred_scores_softmax.shape[0]): 
                query_scores_no_bg = pred_scores_softmax[q_idx, :num_model_classes]
                score, class_idx = torch.max(query_scores_no_bg, dim=0)
                score = score.item()
                class_idx = class_idx.item()

                if score < detection_config_obj.conf_th: 
                    continue
                
                class_name = class_names[class_idx] 
                box_params_7d = pred_boxes_sample[q_idx].numpy()
                
                translation = box_params_7d[0:3].tolist()
                size = box_params_7d[3:6].tolist() 
                yaw_rad = box_params_7d[6]
                rotation_quat = PyQuaternion(axis=[0,0,1], angle=yaw_rad).elements.tolist()
                
                pred_box_dict_for_eval = {
                    "sample_token": sample_token, "translation": translation, "size": size,
                    "rotation": rotation_quat, "velocity": [np.nan, np.nan], 
                    "detection_name": class_name, "detection_score": score, 
                    "attribute_name": "" 
                }
                sample_pred_boxes_for_eval_list.append(pred_box_dict_for_eval)
            
            if sample_token not in all_pred_eval_boxes.boxes: all_pred_eval_boxes.boxes[sample_token] = []
            all_pred_eval_boxes.boxes[sample_token].extend(sample_pred_boxes_for_eval_list)

            # Konvertiere GT-Daten für dieses Sample
            gt_dets_for_sample_raw_list = raw_gt_detections_batch[i] 
            sample_gt_boxes_for_eval_list = []
            for gt_det_dict in gt_dets_for_sample_raw_list:
                class_idx_gt = gt_det_dict['class_label']
                class_name_gt = class_names[class_idx_gt]
                box_params_gt_7d = gt_det_dict['box_world'] 
                translation_gt = box_params_gt_7d[0:3].tolist()
                size_gt = box_params_gt_7d[3:6].tolist()
                yaw_gt = box_params_gt_7d[6]
                rotation_quat_gt = PyQuaternion(axis=[0,0,1], angle=yaw_gt).elements.tolist()
                velocity_gt = gt_det_dict['velocity_world'].tolist()
                gt_box_dict_for_eval = {
                    "sample_token": sample_token, "translation": translation_gt, "size": size_gt,
                    "rotation": rotation_quat_gt, "detection_name": class_name_gt,
                    "detection_score": 1.0, 
                    "velocity": velocity_gt,
                }
                sample_gt_boxes_for_eval_list.append(gt_box_dict_for_eval)

            if sample_token not in all_gt_eval_boxes.boxes: all_gt_eval_boxes.boxes[sample_token] = []
            all_gt_eval_boxes.boxes[sample_token].extend(sample_gt_boxes_for_eval_list)
            
        log_freq_eval_collect = max(1, num_batches // 5 if num_batches > 0 else 1)
        if (batch_idx + 1) % log_freq_eval_collect == 0 or batch_idx == num_batches - 1 : 
            print(f"Evaluation data collection: [{batch_idx+1}/{num_batches}] processed.")
    
    eval_output_dir = os.path.join(cfg.get("training", {}).get("checkpoint_dir", "."), "eval_output_devkit")
    os.makedirs(eval_output_dir, exist_ok=True)

    # Die JSON-Datei mit Vorhersagen wird nicht mehr für den Konstruktor benötigt,
    # aber die DetectionEval-Klasse könnte sie für andere Zwecke intern erzeugen oder erwarten,
    # wenn man die Ergebnisse auf dem Server einreichen wollte.
    # Für die lokale Berechnung der Metriken übergeben wir die EvalBoxes-Objekte direkt.
    # temp_pred_json_path = os.path.join(eval_output_dir, f"predictions_for_eval_{time.strftime('%Y%m%d-%H%M%S')}.json")
    # try:
    #     with open(temp_pred_json_path, 'w') as f:
    #         # Die EvalBoxes-Klasse hat typischerweise eine .serialize()-Methode, um das richtige Format zu erzeugen
    #         # Für das "submission" format:
    #         submission_json = {"meta": {...}, "results": all_pred_eval_boxes.boxes}
    #         json.dump(submission_json, f, indent=2)
    #     print(f"Vorhersage-JSON für DevKit Eval gespeichert unter: {temp_pred_json_path}")
    # except Exception as e:
    #     print(f"FEHLER beim Speichern der Vorhersage-JSON für DevKit Eval: {e}")
    #     # return {"mAP": -1.0, "error": "Failed to save prediction JSON for DevKit Eval"}
    # # Wir fahren ohne die JSON fort, da wir die Objekte direkt übergeben.

    print(f"\nRunning TruckScenes DetectionEval. Output directory for plots: {eval_output_dir}")
    
    # KORRIGIERTER AUFRUF für DetectionEval, der gt_boxes und pred_boxes direkt übergibt
    detection_evaluator = DetectionEval(
        gt_boxes=all_gt_eval_boxes,      
        pred_boxes=all_pred_eval_boxes,  
        cfg=detection_config_obj, 
        eval_set=eval_main_cfg.get("eval_split_name", "mini_val"), 
        output_dir=eval_output_dir,
        verbose=True
        # nusc=ts_instance, # Entfernt, da es den TypeError verursacht hat
        # result_path=temp_pred_json_path # Entfernt, da pred_boxes direkt übergeben wird
    )
    detection_evaluator.run(render_curves=True) 

    print("\n--- Evaluation Results (TruckScenes DevKit) ---")
    final_metrics_to_return = {"mAP": 0.0} 

    if detection_evaluator.metrics: 
        mean_ap = detection_evaluator.metrics.mean_ap
        print(f"Mean AP (mAP) over distance thresholds: {mean_ap:.4f}")
        final_metrics_to_return["mAP"] = mean_ap
        
        if detection_evaluator.metrics.mean_dist_aps: 
            for class_name_eval in detection_config_obj.class_names: 
                if class_name_eval in detection_evaluator.metrics.mean_dist_aps:
                    ref_dist_th_for_print = detection_config_obj.dist_ths[0] if detection_config_obj.dist_ths.size > 0 else -1.0
                    ap_at_ref_dist = detection_evaluator.metrics.mean_dist_aps[class_name_eval].get(ref_dist_th_for_print, -1.0)
                    print(f"  AP for class '{class_name_eval}' @ dist_th={ref_dist_th_for_print}m: {ap_at_ref_dist:.4f}")
                    final_metrics_to_return[f"AP_{class_name_eval}_@{ref_dist_th_for_print}m"] = ap_at_ref_dist
    else:
        print("Evaluation metrics object in DetectionEval is None or empty.")
    
    print(f"Evaluation artifacts (like PR curves) should be in: {eval_output_dir}")
    print("--- Evaluation Finished ---")
    return final_metrics_to_return


def main_train_loop(cli_args: argparse.Namespace):
    cfg = load_config(cli_args.config_path)
    print(f"Configuration loaded from: {cli_args.config_path}")
    if cli_args.device: cfg["training"]["device"] = cli_args.device
    if cli_args.batch_size: cfg["training"]["batch_size"] = cli_args.batch_size
    if cli_args.epochs: cfg["training"]["epochs"] = cli_args.epochs
    if cli_args.lr: cfg["training"]["learning_rate"] = cli_args.lr
    if cli_args.checkpoint_dir: cfg["training"]["checkpoint_dir"] = cli_args.checkpoint_dir

    train_cfg = cfg.get("training", {}); model_cfg = cfg.get("model", {}); loss_cfg = cfg.get("loss", {}); dataset_cfg = cfg.get("dataset", {}); eval_cfg = cfg.get("evaluation", {})
    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    print("Initializing datasets and dataloaders...")
    train_dataset = ObjectFusionGTDataset(dataroot=dataset_cfg["dataroot"], version=dataset_cfg["version"], split_name=train_cfg.get("train_split_name", 'mini_train'), pipeline_config=cfg, verbose=cli_args.verbose_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=train_cfg.get("batch_size", 2), shuffle=True, num_workers=train_cfg.get("num_workers", 0), collate_fn=object_fusion_gt_collate_fn, pin_memory=True if device.type == 'cuda' else False, drop_last=True)
    val_dataset = ObjectFusionGTDataset(dataroot=dataset_cfg["dataroot"], version=dataset_cfg["version"], split_name=train_cfg.get("val_split_name", 'mini_val'), pipeline_config=cfg, verbose=cli_args.verbose_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=train_cfg.get("batch_size", 2), shuffle=False, num_workers=train_cfg.get("num_workers", 0), collate_fn=object_fusion_gt_collate_fn, pin_memory=True if device.type == 'cuda' else False)
    print(f"Train Dataloader: {len(train_dataloader)} batches. Val Dataloader: {len(val_dataloader)} batches.")
    if len(train_dataloader) == 0 or len(val_dataloader) == 0: print("FEHLER: Dataloader leer."); return

    print("Initializing model...")
    encoder = ObjectEncoder(num_input_features=model_cfg.get("num_input_features", 10), d_model=model_cfg.get("d_model", 256), nhead=model_cfg.get("nhead", 8), num_encoder_layers=model_cfg.get("num_encoder_layers", 3), dim_feedforward=model_cfg.get("dim_feedforward_encoder", model_cfg.get("d_model", 256) * 4), dropout=model_cfg.get("dropout", 0.1), pe_max_coord_val=model_cfg.get("pe_max_coord_val", 150.0), pe_num_freq_bands=model_cfg.get("pe_num_freq_bands"))
    decoder = ObjectFusionTransformerDecoder(d_model=model_cfg.get("d_model", 256), nhead=model_cfg.get("nhead", 8), num_decoder_layers=model_cfg.get("num_decoder_layers", 3), dim_feedforward=model_cfg.get("dim_feedforward_decoder", model_cfg.get("d_model", 256) * 4), dropout=model_cfg.get("dropout", 0.1), num_queries=model_cfg.get("num_queries", 100), num_classes=model_cfg.get("num_classes", 28), box_dim=model_cfg.get("box_dim", 7))
    model = ObjectFusionTransformerModel(encoder, decoder).to(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1 and train_cfg.get("use_dataparallel", False): model = nn.DataParallel(model)
    print(f"Modell mit {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainierbaren Parametern initialisiert.")
    
    print("Initializing loss function and optimizer...")
    matcher = HungarianMatcher(cost_class=loss_cfg.get("cost_class_weight", 1.0), cost_bbox_l1=loss_cfg.get("cost_bbox_l1_weight", 5.0), cost_giou_bev=loss_cfg.get("cost_giou_bev_weight", 2.0))
    criterion = SetCriterion(num_classes=model_cfg.get("num_classes", 28), matcher=matcher, eos_coef=loss_cfg.get("eos_coefficient", 0.1), losses=loss_cfg.get("losses_to_compute", ["labels", "boxes_l1"]), weight_dict=loss_cfg.get("loss_weight_dict", {"loss_ce":1.0, "loss_bbox_l1":5.0})).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.get("learning_rate", 1e-4), weight_decay=train_cfg.get("weight_decay", 1e-4))
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg.get("lr_drop_epoch", 40), gamma=train_cfg.get("lr_scheduler_gamma", 0.1)) 
    
    checkpoint_dir = train_cfg.get("checkpoint_dir", "./checkpoints_c_pipeline"); os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf'); start_epoch = 0
    resume_path = cli_args.resume_checkpoint if cli_args.resume_checkpoint else train_cfg.get("resume_checkpoint")
    if resume_path and os.path.isfile(resume_path): start_epoch, best_val_loss, _ = load_checkpoint(resume_path, model, optimizer, device)
    elif train_cfg.get("resume_if_checkpoint_exists", True):
        latest_chkpt = os.path.join(checkpoint_dir, "checkpoint_latest.pth.tar")
        if os.path.isfile(latest_chkpt): start_epoch, best_val_loss, _ = load_checkpoint(latest_chkpt, model, optimizer, device)

    print(f"\n--- Starting Training from Epoch {start_epoch+1} ---")
    num_epochs = train_cfg.get("epochs", 10)
    for epoch in range(start_epoch, num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        train_loss = train_one_epoch(model, criterion, train_dataloader, optimizer, device, epoch, cfg, print_freq=train_cfg.get("print_freq", 10))
        val_loss = validate_one_epoch(model, criterion, val_dataloader, device, epoch, cfg)
        lr_scheduler.step()
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        save_checkpoint(epoch, model, optimizer, val_loss, os.path.join(checkpoint_dir, "checkpoint_latest.pth.tar"), config=cfg)
        if is_best: save_checkpoint(epoch, model, optimizer, best_val_loss, os.path.join(checkpoint_dir, "model_best.pth.tar"), config=cfg); print(f"Saved new best model in Epoch {epoch+1} with Val-Loss {best_val_loss:.4f}")
        save_every = train_cfg.get("save_every_k_epochs",0)
        if save_every > 0 and (epoch+1)%save_every==0 : save_checkpoint(epoch,model,optimizer,val_loss, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth.tar"), config=cfg)
    print("\n--- Training Finished ---")

    print("\n--- Performing Final Evaluation ---")
    # ts_instance wird für die korrigierte DetectionEval-Initialisierung nicht mehr direkt benötigt,
    # aber es schadet nicht, sie hier zu erstellen, falls andere Teile sie benötigen.
    # Für die main-branch API von DetectionEval ist es optional.
    # ts_instance_for_eval = TruckScenes(version=dataset_cfg["version"], dataroot=dataset_cfg["dataroot"], verbose=False)

    best_model_path = os.path.join(checkpoint_dir, "model_best.pth.tar")
    if os.path.isfile(best_model_path):
        print(f"Loading best model from {best_model_path} for final evaluation...")
        _, _, _ = load_checkpoint(best_model_path, model, optimizer=None, device=device) 
    else:
        print("No 'model_best.pth.tar' found. Evaluating with the last trained model state.")
    eval_split = eval_cfg.get("eval_split_name", train_cfg.get("val_split_name", "mini_val"))
    print(f"Evaluating on the '{eval_split}' split.")
    eval_dataloader = val_dataloader
    if eval_split != train_cfg.get("val_split_name", 'mini_val'): 
        print(f"Creating separate Dataloader for evaluation on split: {eval_split}")
        eval_dataset = ObjectFusionGTDataset(dataroot=dataset_cfg["dataroot"], version=dataset_cfg["version"], split_name=eval_split, pipeline_config=cfg, verbose=cli_args.verbose_dataset)
        eval_dataloader = DataLoader(eval_dataset, batch_size=train_cfg.get("batch_size", 2), shuffle=False, num_workers=train_cfg.get("num_workers", 0), collate_fn=object_fusion_gt_collate_fn)
        if len(eval_dataloader) == 0: print(f"FEHLER: Eval-Dataloader für Split '{eval_split}' leer."); return
    
    # Übergebe die ts_instance hier, falls DetectionEval sie doch intern braucht.
    # Für die main-branch API ist es optional, wenn gt_boxes und pred_boxes übergeben werden.
    # Da wir jetzt wieder auf die API umstellen, die gt_boxes und pred_boxes direkt nimmt,
    # ist ts_instance hier nicht mehr zwingend für den DetectionEval-Konstruktor.
    metrics = evaluate_model(model, eval_dataloader, device, cfg, 
                             # ts_instance=ts_instance_for_eval, # Entfernt für diesen Versuch
                             criterion=criterion) 
    print("\nFinal Evaluation Metrics:", metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trainingsskript für ObjectFusionTransformer (C-Module Pipeline)")
    parser.add_argument('--config_path', type=str, default="config/pipeline_c_modules.yaml", help='Pfad zur YAML-Konfigurationsdatei.')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Optionaler Pfad zu einem Checkpoint.')
    parser.add_argument('--verbose_dataset', action='store_true', help='Ausführliche Dataset-Ausgaben.')
    parser.add_argument('--device', type=str, help="Gerät (z.B. 'cuda:0', 'cpu'). Überschreibt Config.")
    parser.add_argument('--epochs', type=int, help="Anzahl Epochen. Überschreibt Config.")
    parser.add_argument('--batch_size', type=int, help="Batch-Größe. Überschreibt Config.")
    parser.add_argument('--lr', type=float, help="Lernrate. Überschreibt Config.")
    parser.add_argument('--checkpoint_dir', type=str, help="Verzeichnis für Checkpoints. Überschreibt Config.")
    
    cli_args = parser.parse_args()

    if not os.path.exists(cli_args.config_path):
        print(f"FEHLER: Konfigurationsdatei '{cli_args.config_path}' nicht gefunden.")
    else:
        main_train_loop(cli_args)
