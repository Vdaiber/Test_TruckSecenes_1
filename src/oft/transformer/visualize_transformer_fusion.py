"""
Visualisierungsskript für den ObjectFusionTransformer (OFT).
- Lädt ein trainiertes Modell und eine Konfiguration.
- Wählt ein Sample aus einem Datensatz-Split aus.
- Führt eine Inferenz durch.
- Rekonstruiert die vorhergesagten Boxen aus den Offsets unter Verwendung des HungarianMatchers 
  und der GT-Referenzboxen (analog zur korrekten Evaluation).
- Zeichnet Ground-Truth (grün) und vorhergesagte (farbige) Boxen auf das entsprechende Kamerabild.

**Version 6: Behebt TypeError, indem eine lokale Zeichenfunktion verwendet wird.**
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import math

import hydra
from omegaconf import DictConfig, OmegaConf, SCMode

# Importe aus unserem Projekt
from oft.data.transformer_gt_dataset import ObjectFusionGTDataset
from oft.data.transformer_gt_collate import object_fusion_gt_collate_fn
from oft.transformer.encoder import ObjectEncoder
from oft.transformer.decoder import ObjectFusionTransformerDecoder
from oft.transformer.loss import SetCriterion, HungarianMatcher
from oft.transformer.train import ObjectFusionTransformerModel, load_checkpoint

# Importe aus dem TruckScenes DevKit und Hilfsfunktionen
from truckscenes import TruckScenes
from truckscenes.utils.data_classes import Box as DevkitBox
from pyquaternion import Quaternion as PyQuaternion
from oft.utils.sensor_utils import get_camera_intrinsic, get_sensor_extrinsic
# NEU: view_points wird für die lokale Zeichenfunktion benötigt
from truckscenes.utils.geometry_utils import view_points

# Globale Farbdefinition für die Visualisierung, basierend auf dem truckscenes-devkit (RGB)
CLASS_COLORS_VIS_RGB = {
    "car": (255, 158, 0), "truck": (255, 99, 71), "bus": (255, 69, 0),
    "trailer": (255, 140, 0), "other_vehicle": (233, 150, 70),
    "pedestrian": (0, 0, 230), "motorcycle": (255, 61, 99), "bicycle": (220, 20, 60),
    "traffic_cone": (47, 79, 79), "barrier": (112, 128, 144),
    "animal": (70, 130, 180), "traffic_sign": (222, 184, 135),
    '__GT__': (50, 255, 150),
}

def _draw_boxes_on_image_local(
    image: np.ndarray,
    boxes: List[DevkitBox], 
    camera_k_matrix: np.ndarray, 
    world_to_sensor_transform: np.ndarray, 
    line_thickness: int = 2,
    z_threshold: float = 0.1,
    is_gt: bool = False
) -> np.ndarray:
    """
    Lokale Kopie der Zeichenfunktion, die direkt die Farben basierend auf dem Box-Namen
    und dem is_gt Flag verwendet.
    """
    img_out = image.copy()
    img_height, img_width = img_out.shape[:2]
    edges = [
        (0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)
    ]
    for box in boxes:
        box_corners_world = box.corners()
        box_corners_world_h = np.vstack((box_corners_world, np.ones((1, 8))))
        box_corners_sensor_h = world_to_sensor_transform @ box_corners_world_h
        box_corners_sensor = box_corners_sensor_h[:3, :]

        if np.all(box_corners_sensor[2, :] <= z_threshold):
            continue

        image_points_raw = view_points(box_corners_sensor, camera_k_matrix, normalize=True)
        image_points = image_points_raw[:2, :].astype(int)

        if is_gt:
            rgb_color = CLASS_COLORS_VIS_RGB['__GT__']
        else:
            rgb_color = CLASS_COLORS_VIS_RGB.get(box.name, (255, 0, 255)) # Fallback Magenta
        
        # Konvertiere RGB zu BGR für OpenCV
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])

        for i, j in edges:
            if box_corners_sensor[2, i] > z_threshold and box_corners_sensor[2, j] > z_threshold:
                p1 = tuple(image_points[:, i])
                p2 = tuple(image_points[:, j])
                cv2.line(img_out, p1, p2, bgr_color, line_thickness, cv2.LINE_AA)
    return img_out

@hydra.main(config_path="../../../config", config_name="pipeline_c_modules.yaml", version_base=None)
def main(cfg: DictConfig):
    logger = logging.getLogger("visualize_logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    if hydra.core.hydra_config.HydraConfig.initialized():
        hydra_logger = logging.getLogger("hydra")
        if hydra_logger:
            hydra_logger.handlers.clear()

    cfg_dict = OmegaConf.to_container(cfg, resolve=True, structured_config_mode=SCMode.DICT_CONFIG)
    
    dataset_cfg = cfg_dict.get("dataset", {})
    model_cfg = cfg_dict.get("model", {})
    vis_cfg = cfg_dict.get("visualization", {})
    training_cfg = cfg_dict.get("training", {})
    render_cfg = cfg_dict.get("render", {})

    device = torch.device(training_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Verwende Gerät: {device}")

    logger.info("Initialisiere TruckScenes und Dataset für Sample-Auswahl...")
    ts = TruckScenes(version=dataset_cfg["version"], dataroot=dataset_cfg["dataroot"], verbose=False)
    
    split_for_sample_access = training_cfg.get("val_split_name", "mini_val")
    logger.info(f"Lade Samples aus dem Split: '{split_for_sample_access}'")
    vis_dataset = ObjectFusionGTDataset(
        dataroot=dataset_cfg["dataroot"], version=dataset_cfg["version"],
        split_name=split_for_sample_access, 
        pipeline_config=cfg_dict, 
        verbose=False
    )

    if not vis_dataset.sample_tokens:
        logger.error(f"Fehler: Keine Samples im Dataset für Split '{split_for_sample_access}' gefunden.")
        return

    sample_to_visualize_token: Optional[str] = None
    sample_idx_for_vis = training_cfg.get('sample_idx_for_vis', 0)
    if 0 <= sample_idx_for_vis < len(vis_dataset):
        sample_to_visualize_token = vis_dataset.sample_tokens[sample_idx_for_vis]
    else:
        logger.error(f"Fehler: sample_idx_for_vis {sample_idx_for_vis} ist außerhalb des gültigen Bereichs für Split '{split_for_sample_access}'.")
        return

    logger.info(f"Visualisiere Sample-Token: {sample_to_visualize_token}")

    try:
        sample_idx_in_dataset = vis_dataset.sample_tokens.index(sample_to_visualize_token)
    except ValueError:
        logger.error(f"Fehler: Sample-Token {sample_to_visualize_token} nicht im geladenen Split '{split_for_sample_access}' gefunden.")
        return

    checkpoint_path = training_cfg.get('resume_checkpoint')
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.error(f"Fehler: Gültiger Checkpoint-Pfad muss über '+training.resume_checkpoint=/path/to/checkpoint' angegeben werden.")
        logger.error(f"Versuchter Pfad (aus Konfig): {checkpoint_path}")
        return
    
    logger.info(f"Lade Modell von Checkpoint: {checkpoint_path}")
    
    encoder_keys = ['d_model', 'nhead', 'num_encoder_layers', 'dim_feedforward_encoder', 'dropout', 'activation', 'pe_max_coord_val', 'num_input_features']
    decoder_keys = ['d_model', 'nhead', 'num_decoder_layers', 'dim_feedforward_decoder', 'dropout', 'activation', 'num_queries', 'num_classes', 'box_dim', 'center_offset_scale']
    
    encoder_params = {k: model_cfg[k] for k in encoder_keys if k in model_cfg}
    decoder_params = {k: model_cfg[k] for k in decoder_keys if k in model_cfg}
    
    if 'dim_feedforward_encoder' in encoder_params:
        encoder_params['dim_feedforward'] = encoder_params.pop('dim_feedforward_encoder')
    if 'dim_feedforward_decoder' in decoder_params:
        decoder_params['dim_feedforward'] = decoder_params.pop('dim_feedforward_decoder')

    encoder = ObjectEncoder(**encoder_params)
    decoder = ObjectFusionTransformerDecoder(**decoder_params)
    model = ObjectFusionTransformerModel(encoder, decoder)
    
    start_epoch, best_metric = load_checkpoint(str(checkpoint_path), model, None, None, None, device, logger)
    logger.info(f"Checkpoint aus Epoche {start_epoch-1} mit best_metric {best_metric:.4f} geladen.")
    model.to(device)
    model.eval()

    single_sample_data_dict = vis_dataset[sample_idx_in_dataset]
    batch_for_model = object_fusion_gt_collate_fn([single_sample_data_dict])
    
    encoder_input_features = batch_for_model["encoder_input_features"].to(device)
    encoder_input_mask = batch_for_model["encoder_input_mask"].to(device)
    encoder_input_xyz_centers = batch_for_model["encoder_input_xyz_centers"].to(device)
    
    gt_target_labels = batch_for_model['gt_target_labels'].to(device)
    gt_target_boxes_log_dims = batch_for_model['gt_target_boxes_log_dims'].to(device)
    gt_target_boxes_actual_dims = batch_for_model['gt_target_boxes_actual_dims'].to(device)

    matcher_cfg = cfg_dict.get('loss', {})
    matcher = HungarianMatcher(
        cost_class=float(matcher_cfg.get('cost_class_weight', 2.0)),
        cost_bbox_l1_offset=float(matcher_cfg.get('cost_bbox_l1_offset_weight', 5.0)),
        cost_giou_bev=float(matcher_cfg.get('cost_giou_bev_weight', 2.0)),
        center_offset_scale_for_matcher_cost=float(model_cfg.get('center_offset_scale', 5.0))
    )

    logger.info("Mache Vorhersagen mit dem geladenen Modell...")
    with torch.no_grad():
        predictions = model(encoder_input_features, encoder_input_xyz_centers, encoder_input_mask)
        
        pred_logits_batch = predictions['pred_logits']
        pred_box_offsets_batch = predictions['pred_box_offsets']
        
        indices = matcher(
            pred_logits_batch, pred_box_offsets_batch, gt_target_labels, 
            gt_target_boxes_actual_dims, gt_target_boxes_log_dims
        )
        
        pred_indices, gt_indices = indices[0]
        
        matched_pred_logits = pred_logits_batch[0, pred_indices]
        matched_pred_offsets = pred_box_offsets_batch[0, pred_indices]
        
        valid_gt_mask_i = batch_for_model['gt_target_valid_mask'][0]
        matched_gt_boxes_actual = gt_target_boxes_actual_dims[0, valid_gt_mask_i][gt_indices]
        matched_gt_boxes_log = gt_target_boxes_log_dims[0, valid_gt_mask_i][gt_indices]

        recon_centers = matched_gt_boxes_actual[:, :3] + matched_pred_offsets[:, :3]
        recon_log_dims = matched_gt_boxes_log[:, 3:6] + matched_pred_offsets[:, 3:6]
        recon_actual_dims = torch.exp(recon_log_dims)
        recon_yaws = matched_gt_boxes_actual[:, 6:7] + matched_pred_offsets[:, 6:7]
        recon_yaws = (recon_yaws + math.pi) % (2 * math.pi) - math.pi

        pred_boxes_reconstructed_vehicle = torch.cat(
            (recon_centers, recon_actual_dims, recon_yaws), dim=-1
        ).cpu().numpy()

    class_names = dataset_cfg.get("class_names")
    score_thresh_for_vis = training_cfg.get("score_threshold_for_vis", 0.3)

    predicted_devkit_boxes: List[DevkitBox] = []
    softmax_scores = F.softmax(matched_pred_logits, dim=-1)
    scores, labels = torch.max(softmax_scores[:, :len(class_names)], dim=-1)

    for i in range(pred_boxes_reconstructed_vehicle.shape[0]):
        if scores[i] < score_thresh_for_vis:
            continue
        
        box_params = pred_boxes_reconstructed_vehicle[i]
        label_idx = labels[i].item()
        detection_name = class_names[label_idx]

        db = DevkitBox(
            center=box_params[:3], size=box_params[3:6],
            orientation=PyQuaternion(axis=[0, 0, 1], angle=box_params[6]),
            name=detection_name, score=scores[i].item(),
            token=f"pred_{sample_to_visualize_token}_{i}"
        )
        predicted_devkit_boxes.append(db)
    
    gt_devkit_boxes: List[DevkitBox] = []
    for gt_det in single_sample_data_dict['gt_detections_raw_world_current_frame']:
        box_world = gt_det['box_7d_world']
        db_gt = DevkitBox(
            center=box_world[:3], size=box_world[3:6],
            orientation=PyQuaternion(axis=[0, 0, 1], angle=box_world[6]),
            name=gt_det['detection_name'], token=gt_det['annotation_token']
        )
        gt_devkit_boxes.append(db_gt)

    logger.info(f"{len(predicted_devkit_boxes)} vorhergesagte Boxen (Score > {score_thresh_for_vis:.2f}) und {len(gt_devkit_boxes)} GT-Boxen gefunden.")

    camera_channel = vis_cfg.get("camera_channel", "CAMERA_LEFT_FRONT")
    sample_rec = ts.get('sample', sample_to_visualize_token)
    cam_token = sample_rec['data'].get(camera_channel)
    if not cam_token:
        logger.error(f"Kamerakanal '{camera_channel}' nicht in Sample {sample_to_visualize_token} gefunden.")
        return
        
    sd_record = ts.get('sample_data', cam_token)
    cs_record = ts.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    ego_pose_record = ts.get('ego_pose', sd_record['ego_pose_token'])
    cam_path = os.path.join(ts.dataroot, sd_record['filename'])
    cam_intrinsics = get_camera_intrinsic(cs_record)
    world_to_sensor_transform = get_sensor_extrinsic(ego_pose_record, cs_record)

    image = cv2.imread(cam_path)
    if image is None:
        logger.error(f"Bild konnte nicht geladen werden: {cam_path}")
        return
    
    ego_translation_world = batch_for_model['ego_translations_world'][0].numpy()
    ego_rotation_world = PyQuaternion(batch_for_model['ego_rotations_world_quat'][0].numpy())
    for box in predicted_devkit_boxes:
        box.rotate(ego_rotation_world)
        box.translate(ego_translation_world)

    # Zeichne Boxen mit der lokalen Funktion
    image_with_gt = _draw_boxes_on_image_local(
        image.copy(), gt_devkit_boxes, camera_k_matrix=cam_intrinsics,
        world_to_sensor_transform=world_to_sensor_transform,
        line_thickness=render_cfg.get("line_thickness", 3) + 1,
        is_gt=True
    )
    
    image_with_all = _draw_boxes_on_image_local(
        image_with_gt, predicted_devkit_boxes, camera_k_matrix=cam_intrinsics,
        world_to_sensor_transform=world_to_sensor_transform,
        line_thickness=render_cfg.get("line_thickness", 2),
        is_gt=False
    )

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir) / "visualizations"
    output_dir.mkdir(exist_ok=True)
    checkpoint_name = Path(str(checkpoint_path)).stem.replace('.pth','').replace('.tar','')
    output_filename = f"vis_{sample_to_visualize_token}_{checkpoint_name}.jpg"
    output_path = output_dir / output_filename
    cv2.imwrite(str(output_path), image_with_all)
    logger.info(f"Visualisierung gespeichert unter: {output_path}")

if __name__ == '__main__':
    main()
