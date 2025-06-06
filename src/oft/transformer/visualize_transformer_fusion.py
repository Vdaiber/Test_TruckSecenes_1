"""
Visualisierungsskript für den ObjectFusionTransformer (OFT).
- Lädt ein trainiertes Modell und eine Konfiguration.
- Wählt ein Sample aus einem Datensatz-Split aus.
- Führt eine Inferenz durch.
- Rekonstruiert die vorhergesagten Boxen aus den Offsets unter Verwendung des HungarianMatchers 
  und der GT-Referenzboxen (analog zur korrekten Evaluation).
- Zeichnet Ground-Truth (grün) und vorhergesagte (blau) Boxen auf das entsprechende Kamerabild.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

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
from oft.utils.sensor_utils import get_camera_intrinsic, get_sensor_extrinsic, draw_boxes_on_image

# Globale Farbdefinition für die Visualisierung
CLASS_COLORS_VIS = {
    'car': (0, 0, 255), 'truck': (0, 0, 128), 'bus': (0, 128, 128),
    'trailer': (0, 255, 128), 'other_vehicle': (128, 128, 0),
    'pedestrian': (255, 0, 0), 'motorcycle': (255, 0, 255), 'bicycle': (255, 255, 0),
    'traffic_cone': (255, 128, 0), 'barrier': (128, 0, 0),
    'animal': (128, 64, 0), 'traffic_sign': (0, 128, 0),
    # Spezielle Farben für unsere Visualisierung
    '__PRED__': (50, 150, 255),  # Hellblau für Vorhersagen
    '__GT__': (50, 255, 150),    # Hellgrün für Ground Truth
}

def get_color(class_name: str, mode: str = 'pred') -> tuple:
    """Holt eine Farbe für eine Klasse, mit Fallback-Farben."""
    if mode.lower() == 'gt':
        return CLASS_COLORS_VIS.get('__GT__')
    return CLASS_COLORS_VIS.get(class_name, CLASS_COLORS_VIS.get('__PRED__'))

def main(cfg: DictConfig):
    logger = logging.getLogger("visualize_logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True, structured_config_mode=SCMode.DICT_CONFIG)
    
    dataset_cfg = cfg_dict.get("dataset", {})
    model_cfg = cfg_dict.get("model", {})
    eval_cfg_params = cfg_dict.get("evaluation", {}).get("eval_detection_cfg", {})
    vis_cfg = cfg_dict.get("visualization", {})
    render_cfg = cfg_dict.get("render", {})
    training_cfg = cfg_dict.get("training", {})

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

    # Wähle ein Sample zur Visualisierung aus
    sample_to_visualize_token: Optional[str] = None
    if training_cfg.get('sample_token_for_vis'):
        sample_to_visualize_token = training_cfg.get('sample_token_for_vis')
    else:
        sample_idx = training_cfg.get('sample_idx_for_vis', 0)
        if 0 <= sample_idx < len(vis_dataset):
            sample_to_visualize_token = vis_dataset.sample_tokens[sample_idx]
        else:
            logger.error(f"Fehler: sample_idx_for_vis {sample_idx} ist außerhalb des gültigen Bereichs für Split '{split_for_sample_access}'.")
            return

    logger.info(f"Visualisiere Sample-Token: {sample_to_visualize_token}")

    # Finde den Index des Tokens im Dataset für __getitem__
    try:
        sample_idx_in_dataset = vis_dataset.sample_tokens.index(sample_to_visualize_token)
    except ValueError:
        logger.error(f"Fehler: Sample-Token {sample_to_visualize_token} nicht im geladenen Split '{split_for_sample_access}' gefunden.")
        return

    # Lade das Modell
    checkpoint_path = training_cfg.get('resume_checkpoint')
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        hydra_run_dir = hydra.core.hydra_config.HydraConfig.get().run.dir
        checkpoint_path = Path(hydra_run_dir) / training_cfg.get('checkpoint_dir') / "oft_c_best.pth.tar"
        logger.info(f"Kein expliziter Checkpoint angegeben, versuche besten Checkpoint aus dem Run-Verzeichnis: {checkpoint_path}")
        if not checkpoint_path.exists():
            logger.error(f"Fehler: Checkpoint-Datei nicht gefunden unter {checkpoint_path}")
            return
    
    logger.info(f"Lade Modell von Checkpoint: {checkpoint_path}")
    encoder = ObjectEncoder(**model_cfg) # Nutze **kwargs für einfache Initialisierung
    decoder = ObjectFusionTransformerDecoder(**model_cfg)
    model = ObjectFusionTransformerModel(encoder, decoder)
    
    load_checkpoint(str(checkpoint_path), model, None, None, None, device, logger)
    model.to(device)
    model.eval()

    # Bereite Model-Input vor
    single_sample_data_dict = vis_dataset[sample_idx_in_dataset]
    batch_for_model = object_fusion_gt_collate_fn([single_sample_data_dict])
    
    # Extrahiere Daten aus dem Batch und sende sie auf das Gerät
    encoder_input_features = batch_for_model["encoder_input_features"].to(device)
    encoder_input_mask = batch_for_model["encoder_input_mask"].to(device)
    encoder_input_xyz_centers = batch_for_model["encoder_input_xyz_centers"].to(device)
    
    gt_target_labels = batch_for_model['gt_target_labels'].to(device)
    gt_target_boxes_log_dims = batch_for_model['gt_target_boxes_log_dims'].to(device)
    gt_target_boxes_actual_dims = batch_for_model['gt_target_boxes_actual_dims'].to(device)

    # Erstelle Matcher, um Zuordnungen zu finden
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
        
        # Rekonstruiere Boxen mit der gleichen Logik wie in der Evaluation
        pred_logits_batch = predictions['pred_logits']
        pred_box_offsets_batch = predictions['pred_box_offsets']
        
        indices = matcher(
            pred_logits_batch, pred_box_offsets_batch, gt_target_labels, 
            gt_target_boxes_actual_dims, gt_target_boxes_log_dims
        )
        
        # Wir visualisieren nur für das erste (und einzige) Sample im Batch
        pred_indices, gt_indices = indices[0]
        
        matched_pred_logits = pred_logits_batch[0, pred_indices]
        matched_pred_offsets = pred_box_offsets_batch[0, pred_indices]
        
        valid_gt_mask_i = batch_for_model['gt_target_valid_mask'][0]
        matched_gt_boxes_actual = gt_target_boxes_actual_dims[0, valid_gt_mask_i][gt_indices]
        matched_gt_boxes_log = gt_target_boxes_log_dims[0, valid_gt_mask_i][gt_indices]

        # Rekonstruktion
        recon_centers = matched_gt_boxes_actual[:, :3] + matched_pred_offsets[:, :3]
        recon_log_dims = matched_gt_boxes_log[:, 3:6] + matched_pred_offsets[:, 3:6]
        recon_actual_dims = torch.exp(recon_log_dims)
        recon_yaws = matched_gt_boxes_actual[:, 6:7] + matched_pred_offsets[:, 6:7]
        recon_yaws = (recon_yaws + math.pi) % (2 * math.pi) - math.pi

        pred_boxes_reconstructed_vehicle = torch.cat(
            (recon_centers, recon_actual_dims, recon_yaws), dim=-1
        ).cpu().numpy()

    # Konvertiere GT und Vorhersagen in DevkitBox-Objekte
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

    logger.info(f"{len(predicted_devkit_boxes)} vorhergesagte Boxen und {len(gt_devkit_boxes)} GT-Boxen gefunden.")

    # Lade Kamerabild und Kalibrierungsdaten
    camera_channel = vis_cfg.get("camera_channel", "CAMERA_FRONT")
    sample_rec = ts.get('sample', sample_to_visualize_token)
    cam_token = sample_rec['data'].get(camera_channel)
    if not cam_token:
        logger.error(f"Kamerakanal '{camera_channel}' nicht in Sample {sample_to_visualize_token} gefunden.")
        return
        
    cam_path, _, cam_intrinsics, ego_pose, cs_rec = ts.get_sample_data(cam_token, get_sensor_extrinsics=True, get_color=True)
    image = cv2.imread(cam_path)
    world_to_sensor_transform = get_sensor_extrinsic(ego_pose, cs_rec)

    # Transformiere rekonstruierte Fahrzeug-Boxen in Welt-Koordinaten für die Visualisierung
    ego_translation_world = batch_for_model['ego_translations_world'][0].numpy()
    ego_rotation_world = PyQuaternion(batch_for_model['ego_rotations_world_quat'][0].numpy())
    for box in predicted_devkit_boxes:
        box.rotate(ego_rotation_world)
        box.translate(ego_translation_world)

    # Zeichne Boxen
    image_with_gt = draw_boxes_on_image(
        image.copy(), gt_devkit_boxes, K=cam_intrinsics,
        world_to_sensor_transform=world_to_sensor_transform,
        line_thickness=2, colors={b.name: get_color(b.name, 'gt') for b in gt_devkit_boxes}
    )
    image_with_all = draw_boxes_on_image(
        image_with_gt, predicted_devkit_boxes, K=cam_intrinsics,
        world_to_sensor_transform=world_to_sensor_transform,
        line_thickness=2, colors={b.name: get_color(b.name, 'pred') for b in predicted_devkit_boxes}
    )

    # Speichere das Ergebnis
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir) / "visualizations"
    output_dir.mkdir(exist_ok=True)
    checkpoint_name = Path(str(checkpoint_path)).stem
    output_filename = f"vis_{sample_to_visualize_token}_{checkpoint_name}.jpg"
    output_path = output_dir / output_filename
    cv2.imwrite(str(output_path), image_with_all)
    logger.info(f"Visualisierung gespeichert unter: {output_path}")

if __name__ == '__main__':
    # Dieses Skript wird am besten über Hydra aufgerufen, damit die Konfiguration korrekt geladen wird.
    # Beispielhafter Hydra-Aufruf:
    # python src/oft/transformer/visualize_transformer_fusion.py \
    #   training.resume_checkpoint="/path/to/your/oft_c_best.pth.tar" \
    #   training.sample_idx_for_vis=10 
    
    # Temporäre Fallback-Logik, um das Skript ohne Hydra lauffähig zu machen (eingeschränkt)
    if not hydra.core.hydra_config.HydraConfig.initialized():
        print("WARNUNG: Hydra nicht initialisiert. Versuche, die Konfiguration manuell zu laden. "
              "Dies ist nur für einfaches Debugging gedacht.")
        
        # Manuelle Konfiguration für den Notfall
        default_config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'pipeline_c_modules.yaml')
        if not os.path.exists(default_config_path):
             raise FileNotFoundError(f"Manuelle Konfiguration konnte nicht geladen werden: {default_config_path}")

        # Erstelle ein minimales OmegaConf-Objekt für den Test
        cfg_obj = OmegaConf.load(default_config_path)

        # Überschreibe manuell Parameter, die normalerweise per Kommandozeile kommen würden
        # HINWEIS: Passe diese Pfade für deinen lokalen Test an!
        OmegaConf.update(cfg_obj, "training.resume_checkpoint", "/app/output/hydra_runs/oft_c_offset_norm_v1/2025-06-06/10-20-03/checkpoints_oft_c_offset_norm_v1/oft_c_best.pth.tar")
        OmegaConf.update(cfg_obj, "training.sample_idx_for_vis", 5) # Wähle ein interessantes Sample

        # Führe die main-Funktion mit der manuell erstellten Konfiguration aus
        # Erstelle eine Dummy-Hydra-Umgebung für den Pfad
        from hydra.core.hydra_config import HydraConfig
        from hydra.core.utils import setup_globals
        setup_globals()
        HydraConfig.instance().set_config(cfg_obj)

        main(cfg_obj)
    else:
        # Normaler Hydra-Start
        @hydra.main(config_path="../../../config", config_name="pipeline_c_modules.yaml", version_base=None)
        def hydra_entry_point(cfg: DictConfig):
            main(cfg)
        
        hydra_entry_point()