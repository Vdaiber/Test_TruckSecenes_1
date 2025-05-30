# src/oft/transformer/visualize_transformer_fusion.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import os
from typing import List, Dict, Any, Optional

# Importe aus unserem Projekt
from oft.utils.config import load_config
from oft.data.transformer_gt_dataset import ObjectFusionGTDataset # Für GT-Daten und Sample-Auswahl
from oft.data.transformer_gt_collate import object_fusion_gt_collate_fn # Um Modell-Input zu erzeugen
from oft.transformer.encoder import ObjectEncoder
from oft.transformer.decoder import ObjectFusionTransformerDecoder
from oft.transformer.train import ObjectFusionTransformerModel, load_checkpoint # Für Modell und Checkpoint-Laden

# Importe aus dem TruckScenes DevKit
from truckscenes import TruckScenes
from truckscenes.utils.data_classes import Box as DevkitBox # Für die Darstellung
# from truckscenes.utils.geometry_utils import BoxVisibility # Nicht direkt hier benötigt, aber gut zu wissen
from pyquaternion import Quaternion as PyQuaternion

# Importiere die passende Visualisierungsfunktion (die Weltkoordinaten-Boxen nimmt)
from oft.utils.sensor_utils import get_camera_intrinsic, get_sensor_extrinsic, draw_boxes_on_image, CLASS_COLORS

def convert_predictions_to_devkit_boxes(
    pred_logits: torch.Tensor, # (NumQueries, NumClasses + 1)
    pred_boxes_7d: torch.Tensor, # (NumQueries, BoxDim)
    class_names: List[str],
    score_threshold: float = 0.3,
    sample_token: str = "default_token" 
) -> List[DevkitBox]:
    """
    Konvertiert die rohen Modellvorhersagen (Logits und Boxen) eines einzelnen Samples
    in eine Liste von truckscenes.utils.data_classes.Box Objekten für die Visualisierung.
    Filtert nach Konfidenz-Score.
    """
    if pred_logits.numel() == 0 or pred_boxes_7d.numel() == 0:
        return []

    pred_scores_softmax = F.softmax(pred_logits, dim=-1) 
    
    num_model_classes = pred_scores_softmax.shape[-1] - 1 
    
    max_scores_no_bg, pred_class_indices_no_bg = torch.max(pred_scores_softmax[:, :num_model_classes], dim=1)

    devkit_boxes: List[DevkitBox] = []
    for i in range(max_scores_no_bg.shape[0]): 
        score = max_scores_no_bg[i].item()
        
        if score < score_threshold:
            continue
            
        class_idx = pred_class_indices_no_bg[i].item()
        if class_idx >= len(class_names): 
            print(f"Warnung: Vorhergesagter Klassenindex {class_idx} außerhalb des Bereichs von class_names (Länge {len(class_names)}).")
            continue
        detection_name = class_names[class_idx]
        
        box_params = pred_boxes_7d[i].cpu().numpy() 
        
        center = box_params[0:3]
        size = box_params[3:6] # w,l,h
        yaw = box_params[6]
        orientation = PyQuaternion(axis=[0, 0, 1], angle=yaw)
        
        db = DevkitBox(
            center=center.tolist(), size=size.tolist(), orientation=orientation,
            name=detection_name, score=score, token=f"pred_{sample_token}_{i}"
        )
        devkit_boxes.append(db)
        
    return devkit_boxes

def main(args):
    print(f"Lade Konfiguration von: {args.config_path}")
    cfg = load_config(args.config_path)

    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    eval_cfg_params = cfg.get("evaluation", {}).get("eval_detection_cfg", {})
    vis_cfg = cfg.get("visualization", {})
    render_cfg = cfg.get("render", {})

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Verwende Gerät: {device}")

    print("Initialisiere TruckScenes und Dataset für Sample-Auswahl...")
    ts = TruckScenes(version=dataset_cfg["version"], dataroot=dataset_cfg["dataroot"], verbose=False)
    
    # Nutze einen Split aus der Trainingskonfiguration, um auf Samples zugreifen zu können
    # (z.B. den Validierungssplit)
    default_split_for_sample_access = cfg.get("training", {}).get("val_split_name", "mini_val")
    split_for_sample_access = args.split if args.split else default_split_for_sample_access

    vis_dataset = ObjectFusionGTDataset(
        dataroot=dataset_cfg["dataroot"], version=dataset_cfg["version"],
        split_name=split_for_sample_access, 
        pipeline_config=cfg, 
        verbose=False # Weniger Output für reine Visualisierung
    )

    if not vis_dataset.sample_tokens:
        print(f"Fehler: Keine Samples im Dataset für Split '{split_for_sample_access}' gefunden.")
        return

    sample_to_visualize_token: Optional[str] = None
    if args.sample_token:
        # Überprüfe, ob der Token im gesamten Datensatz (via ts) existiert, nicht nur im Split
        try:
            ts.get('sample', args.sample_token) # Löst Fehler aus, wenn Token nicht existiert
            sample_to_visualize_token = args.sample_token
        except KeyError:
            print(f"Fehler: Sample-Token '{args.sample_token}' nicht im Datensatz gefunden.")
            return
    elif args.sample_idx is not None:
        if 0 <= args.sample_idx < len(vis_dataset.sample_tokens): # Index bezogen auf den geladenen Split
            sample_to_visualize_token = vis_dataset.sample_tokens[args.sample_idx]
        else:
            print(f"Fehler: Sample-Index {args.sample_idx} ist außerhalb des Bereichs (0-{len(vis_dataset.sample_tokens)-1}) für Split '{split_for_sample_access}'.")
            return
    else: 
        sample_to_visualize_token = vis_dataset.sample_tokens[0] # Nimm das erste Sample des Splits
    
    print(f"Visualisiere Sample-Token: {sample_to_visualize_token}")
    
    # Finde den Index des Tokens im Dataset (für __getitem__)
    try:
        sample_idx_in_dataset = vis_dataset.sample_tokens.index(sample_to_visualize_token)
    except ValueError:
        # Fallback: Wenn Token nicht im Split ist, versuche ihn direkt aus dem Dataset zu laden
        # (Dies erfordert, dass das Dataset so angepasst wird, dass es auch direkte Token-Anfragen unterstützt,
        # oder wir holen die Daten für das Modell-Input anders)
        # Für jetzt: Breche ab, wenn Token nicht im geladenen Split ist, um es einfach zu halten.
        print(f"Fehler: Sample-Token {sample_to_visualize_token} nicht im geladenen Split '{split_for_sample_access}' des Datasets gefunden.")
        print("Tipp: Verwende einen Sample-Token, der im Split enthalten ist, oder gib einen Split an, der ihn enthält.")
        return


    print(f"Lade Modell von Checkpoint: {args.checkpoint_path}")
    encoder = ObjectEncoder(
        num_input_features=model_cfg.get("num_input_features", 10),
        d_model=model_cfg.get("d_model", 256), nhead=model_cfg.get("nhead", 8),
        num_encoder_layers=model_cfg.get("num_encoder_layers", 3),
        dim_feedforward=model_cfg.get("dim_feedforward_encoder", model_cfg.get("d_model", 256) * 4),
        dropout=model_cfg.get("dropout", 0.1),
        pe_max_coord_val=model_cfg.get("pe_max_coord_val", 150.0),
        pe_num_freq_bands=model_cfg.get("pe_num_freq_bands"))
    decoder = ObjectFusionTransformerDecoder(
        d_model=model_cfg.get("d_model", 256), nhead=model_cfg.get("nhead", 8),
        num_decoder_layers=model_cfg.get("num_decoder_layers", 3),
        dim_feedforward=model_cfg.get("dim_feedforward_decoder", model_cfg.get("d_model", 256) * 4),
        dropout=model_cfg.get("dropout", 0.1), num_queries=model_cfg.get("num_queries", 100),
        num_classes=model_cfg.get("num_classes", 28), box_dim=model_cfg.get("box_dim", 7))
    model = ObjectFusionTransformerModel(encoder, decoder)
    
    if not os.path.isfile(args.checkpoint_path):
        print(f"Fehler: Checkpoint-Datei nicht gefunden: {args.checkpoint_path}")
        return
    _, _, _ = load_checkpoint(args.checkpoint_path, model, optimizer=None, device=device)
    model.to(device)
    model.eval()

    # Hole das einzelne Sample aus dem Dataset und erstelle einen Batch der Größe 1
    single_sample_data_dict = vis_dataset[sample_idx_in_dataset]
    # WICHTIG für Eval: `gt_detections_list_raw` muss in Collate-Fn hinzugefügt werden
    # Für Visualisierung hier fügen wir es manuell hinzu, falls nicht von Collate-Fn schon gemacht
    if 'gt_detections_list_raw' not in single_sample_data_dict: # sollte nicht passieren, wenn collate angepasst ist
        single_sample_data_dict_for_collate = {k:v for k,v in single_sample_data_dict.items()}
        single_sample_data_dict_for_collate['gt_detections_list_raw'] = [single_sample_data_dict['gt_detections']]
    else:
        single_sample_data_dict_for_collate = single_sample_data_dict

    batch_for_model = object_fusion_gt_collate_fn([single_sample_data_dict_for_collate])


    encoder_input_features = batch_for_model["encoder_input_features"].to(device)
    encoder_input_mask = batch_for_model["encoder_input_mask"].to(device)
    src_xyz_centers = encoder_input_features[:, :, :3].clone()

    print("Mache Vorhersagen mit dem geladenen Modell...")
    with torch.no_grad():
        outputs_dict = model(
            src_features=encoder_input_features,
            src_padding_mask=encoder_input_mask,
            src_xyz_centers=src_xyz_centers
        )
    
    pred_logits_sample = outputs_dict['pred_logits'][0].cpu() 
    pred_boxes_sample = outputs_dict['pred_boxes'][0].cpu()   

    class_names = dataset_cfg.get("class_names")
    if not class_names:
        print("Fehler: 'dataset.class_names' nicht in Config gefunden.")
        return
    
    score_thresh_for_vis = args.score_threshold if args.score_threshold is not None \
                           else eval_cfg_params.get("conf_th_eval", 0.3)
    predicted_devkit_boxes = convert_predictions_to_devkit_boxes(
        pred_logits_sample, pred_boxes_sample, class_names, score_thresh_for_vis, sample_to_visualize_token
    )
    print(f"{len(predicted_devkit_boxes)} vorhergesagte Boxen nach Score-Filterung (>{score_thresh_for_vis:.2f}).")

    gt_devkit_boxes: List[DevkitBox] = []
    for gt_det in single_sample_data_dict['gt_detections']:
        box_params_gt = gt_det['box_world']
        db_gt = DevkitBox(
            center=box_params_gt[0:3].tolist(), size=box_params_gt[3:6].tolist(),
            orientation=PyQuaternion(axis=[0,0,1], angle=box_params_gt[6]),
            name=class_names[gt_det['class_label']], token=gt_det['annotation_token']
        )
        gt_devkit_boxes.append(db_gt)
    print(f"{len(gt_devkit_boxes)} Ground-Truth Boxen geladen.")

    camera_channel_to_render = args.camera_channel if args.camera_channel else vis_cfg.get("camera_channel", "CAMERA_LEFT_FRONT")
    sample_record_ts = ts.get('sample', sample_to_visualize_token)
    
    if camera_channel_to_render not in sample_record_ts['data']:
        print(f"Fehler: Kamera-Kanal '{camera_channel_to_render}' nicht in Sample {sample_to_visualize_token}. Verfügbar: {list(sample_record_ts['data'].keys())}")
        return

    sd_token = sample_record_ts['data'][camera_channel_to_render]
    sd_record = ts.get('sample_data', sd_token)
    image_path = os.path.join(ts.dataroot, sd_record['filename'])
    image = cv2.imread(image_path)
    if image is None: print(f"Fehler: Bild konnte nicht geladen werden: {image_path}"); return

    cs_record = ts.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    ego_pose_rec = ts.get('ego_pose', sd_record['ego_pose_token'])
    K_matrix = get_camera_intrinsic(cs_record)
    H_world_to_sensor = get_sensor_extrinsic(ego_pose_rec, cs_record)
    
    # Temporäre Kopie der CLASS_COLORS für diese Visualisierung, um globale Änderungen zu vermeiden
    current_class_colors = CLASS_COLORS.copy()
    # GT-Boxen in Grün
    for gt_box in gt_devkit_boxes: gt_box.name = "gt_viz_c6" 
    current_class_colors["gt_viz_c6"] = (0, 255, 0) # Grün für GT
    
    # Vorhergesagte Boxen: Farben basierend auf ihrer echten Klasse (aus class_names)
    # Die 'draw_boxes_on_image' Funktion verwendet box.name, um die Farbe aus CLASS_COLORS zu holen.
    # Die Namen in predicted_devkit_boxes sind bereits die Klassennamen.

    image_to_draw_on = image.copy()
    # Zeichne zuerst GT
    image_to_draw_on = draw_boxes_on_image(
        image=image_to_draw_on, boxes=gt_devkit_boxes, camera_k_matrix=K_matrix,
        world_to_sensor_transform=H_world_to_sensor,
        line_thickness=render_cfg.get("line_thickness", 2),
        z_threshold=render_cfg.get("z_threshold", 0.1),
        box_type_for_debug="gt_c6_viz"
    )
    # Dann zeichne Vorhersagen
    image_to_draw_on = draw_boxes_on_image(
        image=image_to_draw_on, boxes=predicted_devkit_boxes, camera_k_matrix=K_matrix,
        world_to_sensor_transform=H_world_to_sensor,
        line_thickness=render_cfg.get("line_thickness", 2),
        z_threshold=render_cfg.get("z_threshold", 0.1),
        box_type_for_debug="pred_c6_viz"
    )
    
    output_dir = args.output_dir if args.output_dir else os.path.join(cfg.get("training", {}).get("checkpoint_dir", "./checkpoints_temp"), "c6_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    confidence_str = str(score_thresh_for_vis).replace('.', '_')
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
    output_filename = f"transformer_fusion_{sample_to_visualize_token}_{camera_channel_to_render}_{checkpoint_name}_conf{confidence_str}.jpg"
    output_path = os.path.join(output_dir, output_filename)
    
    cv2.imwrite(output_path, image_to_draw_on)
    print(f"Visualisierung gespeichert unter: {output_path}")

    if args.show:
        cv2.imshow(f"Transformer Fusion: {sample_to_visualize_token} | Checkpoint: {checkpoint_name}", image_to_draw_on)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualisiere Transformer Fusionsergebnisse (C6).")
    parser.add_argument('--config_path', type=str, required=True, help='Pfad zur YAML-Konfigurationsdatei (pipeline_c_modules.yaml).')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Pfad zum trainierten Modell-Checkpoint (.pth.tar).')
    parser.add_argument('--sample_token', type=str, default=None, help='Spezieller Sample-Token zur Visualisierung.')
    parser.add_argument('--sample_idx', type=int, default=None, help='Index des Samples im gewählten Split (wenn kein Token gegeben). Default: 0. Split wird aus Config genommen.')
    parser.add_argument('--split', type=str, default=None, help="Name des Splits, aus dem das Sample (--sample_idx) genommen wird (z.B. 'mini_val'). Überschreibt Config für Sample-Auswahl.")
    parser.add_argument('--camera_channel', type=str, default=None, help='Kamerakanal für die Visualisierung (überschreibt Config).')
    parser.add_argument('--device', type=str, default=None, help='Gerät für Inferenz (z.B. "cuda", "cpu"; überschreibt Config).')
    parser.add_argument('--output_dir', type=str, default=None, help='Optionales Verzeichnis zum Speichern der Bilder.')
    parser.add_argument('--score_threshold', type=float, default=None, help="Konfidenz-Schwellenwert für anzuzeigende Vorhersagen (überschreibt Eval-Config).")
    parser.add_argument('--show', action='store_true', help="Zeige das Bild interaktiv an (benötigt Desktop-Umgebung).")
    
    cli_args = parser.parse_args()
    
    if cli_args.sample_idx is None and cli_args.sample_token is None:
        cli_args.sample_idx = 0 # Default auf erstes Sample im Split, wenn nichts angegeben
        print(f"Weder --sample_token noch --sample_idx angegeben. Verwende Index 0 aus dem Split '{cli_args.split if cli_args.split else 'aus Config'}'.")

    main(cli_args)
