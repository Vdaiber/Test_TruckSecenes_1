import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional, Tuple
import numpy as np
import math # Für math.pi

import os
from oft.utils.config import load_config # Für den Test-Block
from omegaconf import OmegaConf # Für die Konvertierung von DictConfig und Erstellung

# --- Hilfsfunktionen für BEV GIoU ---

def get_bev_corners_pytorch(boxes_7d: torch.Tensor) -> torch.Tensor:
    """
    Berechnet die 4 BEV-Eckpunkte für einen Satz von 7D-Boxen.
    Args:
        boxes_7d: Tensor der Form (N, 7) mit [cx, cy, cz, width, length, height, yaw].
                  width (dim 3) ist entlang der lokalen y-Achse der Box.
                  length (dim 4) ist entlang der lokalen x-Achse der Box.
    Returns:
        Tensor der Form (N, 4, 2) mit den Eckpunkten (x,y) in der BEV-Ebene.
        Reihenfolge der Ecken: vorne-links, vorne-rechts, hinten-rechts, hinten-links
                               (relativ zur Box-Orientierung, x-Achse zeigt nach vorne)
    """
    if boxes_7d.ndim == 1:
        boxes_7d = boxes_7d.unsqueeze(0)
    
    # cx, cy, cz, w, l, h, yaw
    centers_x = boxes_7d[:, 0]
    centers_y = boxes_7d[:, 1]
    # cz wird für BEV ignoriert
    widths = boxes_7d[:, 3]  # Entlang der lokalen y-Achse der Box
    lengths = boxes_7d[:, 4] # Entlang der lokalen x-Achse der Box
    # height wird für BEV ignoriert
    yaws = boxes_7d[:, 6]

    # Halbe Länge und Breite
    half_lengths = lengths / 2.0
    half_widths = widths / 2.0

    # Eckpunkte im lokalen Koordinatensystem der Box (x zeigt nach vorne, y nach links)
    # Reihenfolge: vorne-links, vorne-rechts, hinten-rechts, hinten-links
    # (l/2, w/2), (l/2, -w/2), (-l/2, -w/2), (-l/2, w/2)
    corners_local = torch.zeros((boxes_7d.shape[0], 4, 2), device=boxes_7d.device)
    corners_local[:, 0, 0] = half_lengths
    corners_local[:, 0, 1] = half_widths
    corners_local[:, 1, 0] = half_lengths
    corners_local[:, 1, 1] = -half_widths
    corners_local[:, 2, 0] = -half_lengths
    corners_local[:, 2, 1] = -half_widths
    corners_local[:, 3, 0] = -half_lengths
    corners_local[:, 3, 1] = half_widths

    # Rotationsmatrix erstellen
    cos_yaw = torch.cos(yaws)
    sin_yaw = torch.sin(yaws)

    # Rotiere die Eckpunkte
    # x_rot = x_local * cos_yaw - y_local * sin_yaw
    # y_rot = x_local * sin_yaw + y_local * cos_yaw
    corners_rotated_x = corners_local[..., 0] * cos_yaw.unsqueeze(1) - corners_local[..., 1] * sin_yaw.unsqueeze(1)
    corners_rotated_y = corners_local[..., 0] * sin_yaw.unsqueeze(1) + corners_local[..., 1] * cos_yaw.unsqueeze(1)
    
    corners_rotated = torch.stack((corners_rotated_x, corners_rotated_y), dim=-1)

    # Translatiere zu den Weltkoordinaten-Zentren
    corners_world = corners_rotated + torch.stack((centers_x, centers_y), dim=-1).unsqueeze(1)
    
    return corners_world # (N, 4, 2)

def calculate_bev_iou_from_corners_pytorch(
    corners1: torch.Tensor, # (N, 4, 2)
    corners2: torch.Tensor, # (M, 4, 2)
    areas1: torch.Tensor,   # (N,) wahre Flächen (width * length)
    areas2: torch.Tensor    # (M,) wahre Flächen (width * length)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Berechnet eine angenäherte BEV IoU basierend auf den Achsen-parallelen Bounding Boxes (AABB)
    der rotierten Eckpunkte.
    Args:
        corners1: Eckpunkte der ersten Boxen (N, 4, 2).
        corners2: Eckpunkte der zweiten Boxen (M, 4, 2).
        areas1: Tatsächliche Flächen der ersten Boxen (N,).
        areas2: Tatsächliche Flächen der zweiten Boxen (M,).
    Returns:
        Tuple: (iou_matrix (N, M), intersection_area_matrix (N, M))
    """
    N = corners1.shape[0]
    M = corners2.shape[0]
    iou_matrix = torch.zeros((N, M), device=corners1.device)
    intersection_area_matrix = torch.zeros((N, M), device=corners1.device)

    # AABBs für jede Box aus ihren Ecken berechnen
    min_xy1, _ = torch.min(corners1, dim=1) # (N, 2)
    max_xy1, _ = torch.max(corners1, dim=1) # (N, 2)
    
    min_xy2, _ = torch.min(corners2, dim=1) # (M, 2)
    max_xy2, _ = torch.max(corners2, dim=1) # (M, 2)

    for i in range(N):
        for j in range(M):
            # Schnittfläche der AABBs der Ecken
            inter_min_x = torch.max(min_xy1[i, 0], min_xy2[j, 0])
            inter_min_y = torch.max(min_xy1[i, 1], min_xy2[j, 1])
            inter_max_x = torch.min(max_xy1[i, 0], max_xy2[j, 0])
            inter_max_y = torch.min(max_xy1[i, 1], max_xy2[j, 1])

            inter_width = torch.clamp(inter_max_x - inter_min_x, min=0.0)
            inter_height = torch.clamp(inter_max_y - inter_min_y, min=0.0)
            intersection_aabb_corners = inter_width * inter_height
            intersection_area_matrix[i, j] = intersection_aabb_corners

            # Union basierend auf wahren Boxflächen und AABB-Schnittfläche der Ecken
            union_area = areas1[i] + areas2[j] - intersection_aabb_corners
            if union_area > 1e-6: # Numerische Stabilität
                iou_matrix[i, j] = intersection_aabb_corners / union_area
            else:
                iou_matrix[i, j] = torch.tensor(0.0, device=corners1.device)
    
    return torch.clamp(iou_matrix, min=0.0, max=1.0), intersection_area_matrix


def calculate_enclosing_box_area_bev_pytorch(
    corners1: torch.Tensor, # (N, 4, 2)
    corners2: torch.Tensor  # (M, 4, 2)
) -> torch.Tensor:
    """
    Berechnet die Fläche der kleinsten achsenparallelen Bounding Box (AABB),
    die jeweils ein Paar von Boxen (repräsentiert durch ihre Eckpunkte) umschließt.
    Args:
        corners1: Eckpunkte der ersten Boxen (N, 4, 2).
        corners2: Eckpunkte der zweiten Boxen (M, 4, 2).
    Returns:
        Tensor der Form (N, M) mit den Flächen der umschließenden AABBs.
    """
    N = corners1.shape[0]
    M = corners2.shape[0]
    enclosing_area_matrix = torch.zeros((N, M), device=corners1.device)

    for i in range(N):
        for j in range(M):
            # Kombiniere die Eckpunkte beider Boxen
            all_corners = torch.cat((corners1[i], corners2[j]), dim=0) # (8, 2)
            
            min_coords, _ = torch.min(all_corners, dim=0) # (2,) -> [x_min, y_min]
            max_coords, _ = torch.max(all_corners, dim=0) # (2,) -> [x_max, y_max]
            
            enclosing_width = max_coords[0] - min_coords[0]
            enclosing_height = max_coords[1] - min_coords[1]
            enclosing_area_matrix[i, j] = enclosing_width * enclosing_height
            
    return enclosing_area_matrix

def generalized_bev_iou_pytorch(boxes1_7d: torch.Tensor, boxes2_7d: torch.Tensor) -> torch.Tensor:
    """
    Berechnet den Generalisierten BEV IoU (GIoU) zwischen zwei Sätzen von 7D-Boxen.
    Verwendet eine angenäherte IoU basierend auf AABBs der rotierten Ecken.
    Args:
        boxes1_7d: Tensor der Form (N, 7) [cx, cy, cz, w, l, h, yaw].
        boxes2_7d: Tensor der Form (M, 7) [cx, cy, cz, w, l, h, yaw].
    Returns:
        Tensor der Form (N, M) mit den paarweisen GIoU-Werten.
    """
    eps = 1e-7 # Für numerische Stabilität

    # Wahre Flächen der Boxen (width * length)
    areas1 = boxes1_7d[:, 3] * boxes1_7d[:, 4] # (N,)
    areas2 = boxes2_7d[:, 3] * boxes2_7d[:, 4] # (M,)

    # Eckpunkte im BEV
    corners1 = get_bev_corners_pytorch(boxes1_7d) # (N, 4, 2)
    corners2 = get_bev_corners_pytorch(boxes2_7d) # (M, 4, 2)

    # Angenäherte IoU und Schnittfläche
    iou_approx, intersection_area_approx = calculate_bev_iou_from_corners_pytorch(corners1, corners2, areas1, areas2) # (N,M), (N,M)

    # Fläche der umschließenden Box (Area_C)
    enclosing_area = calculate_enclosing_box_area_bev_pytorch(corners1, corners2) # (N,M)

    # Union Area
    union_area = areas1.unsqueeze(1) + areas2.unsqueeze(0) - intersection_area_approx # (N,M)
    union_area = torch.clamp(union_area, min=eps) # Verhindere Division durch Null

    # GIoU berechnen
    giou = iou_approx - (enclosing_area - union_area) / (enclosing_area + eps)
    
    return torch.clamp(giou, min=-1.0, max=1.0) # GIoU ist im Bereich [-1, 1]


class HungarianMatcher(nn.Module):
    """
    Matcher, der Vorhersagen den Ground-Truth-Boxen zuordnet.
    Basierend auf dem DETR-Matcher.
    """
    def __init__(self,
                 cost_class: float = 1.0,
                 cost_bbox_l1: float = 1.0,
                 cost_giou_bev: float = 1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox_l1 = cost_bbox_l1
        self.cost_giou_bev = cost_giou_bev
        if cost_class == 0 and cost_bbox_l1 == 0 and cost_giou_bev == 0:
            raise ValueError("Alle Kosten-Gewichte im Matcher dürfen nicht null sein.")

    @torch.no_grad()
    def forward(self,
                pred_logits: torch.Tensor,  # (Batch, NumQueries, NumClasses + 1)
                pred_boxes: torch.Tensor,   # (Batch, NumQueries, BoxDim=7)
                gt_labels_b: torch.Tensor,    # (Batch, MaxNumGTObjects) - gepadded mit -1
                gt_boxes_b: torch.Tensor      # (Batch, MaxNumGTObjects, BoxDim=7) - gepadded
               ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, num_queries = pred_logits.shape[:2]
        indices = []

        for i in range(batch_size):
            valid_gt_mask_i = (gt_labels_b[i] >= 0)
            if not valid_gt_mask_i.any():
                indices.append((torch.tensor([], dtype=torch.long, device=pred_logits.device),
                                torch.tensor([], dtype=torch.long, device=pred_logits.device)))
                continue

            current_gt_labels_i = gt_labels_b[i][valid_gt_mask_i]
            current_gt_boxes_i = gt_boxes_b[i][valid_gt_mask_i]
            num_gt_i = current_gt_labels_i.shape[0]

            if num_gt_i == 0:
                indices.append((torch.tensor([], dtype=torch.long, device=pred_logits.device),
                                torch.tensor([], dtype=torch.long, device=pred_logits.device)))
                continue

            current_pred_logits_i = pred_logits[i]
            current_pred_boxes_i = pred_boxes[i]

            prob = current_pred_logits_i.softmax(-1)
            cost_class_matrix = -prob[:, current_gt_labels_i]

            cost_bbox_l1_matrix = torch.cdist(current_pred_boxes_i, current_gt_boxes_i, p=1)
            
            # Verwende den neuen GIoU-Loss für die Kostenberechnung
            # GIoU-Werte sind im Bereich [-1, 1]. Kosten sollten positiv sein und kleiner für bessere Übereinstimmung.
            # Daher verwenden wir 1 - GIoU als Kosten (Bereich [0, 2]).
            giou_values = generalized_bev_iou_pytorch(current_pred_boxes_i, current_gt_boxes_i) # (NumQueries, NumGT_i)
            cost_giou_matrix = 1.0 - giou_values # Kleinere Kosten für höhere GIoU

            C = (self.cost_class * cost_class_matrix +
                 self.cost_bbox_l1 * cost_bbox_l1_matrix +
                 self.cost_giou_bev * cost_giou_matrix)
            
            C_np = C.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(C_np)
            indices.append((torch.as_tensor(row_ind, dtype=torch.long, device=pred_logits.device),
                            torch.as_tensor(col_ind, dtype=torch.long, device=pred_logits.device)))
        return indices


class SetCriterion(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 matcher: HungarianMatcher,
                 eos_coef: float,
                 losses: List[str],
                 weight_dict: Dict[str, float]
                ): 
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef 
        self.losses = losses
        self.weight_dict = weight_dict

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def _get_src_permutation_idx(self, indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_labels(self, 
                    pred_logits: torch.Tensor,
                    gt_labels_b: torch.Tensor,
                    indices: List[Tuple[torch.Tensor, torch.Tensor]],
                    num_total_boxes: int
                   ) -> Dict[str, torch.Tensor]:
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes,
                                    dtype=torch.long, device=pred_logits.device)
        
        batch_indices_for_preds, matched_pred_indices_batch = self._get_src_permutation_idx(indices)
        
        target_classes_o_list = [] 
        for i, (_, gt_idx_sample) in enumerate(indices):
            valid_gt_labels_sample_i = gt_labels_b[i][gt_labels_b[i] >= 0]
            if gt_idx_sample.numel() > 0 and valid_gt_labels_sample_i.numel() > 0:
                 if gt_idx_sample.max() < len(valid_gt_labels_sample_i):
                    target_classes_o_list.append(valid_gt_labels_sample_i[gt_idx_sample])

        if target_classes_o_list:
            target_classes_o = torch.cat(target_classes_o_list)
            if target_classes_o.numel() == batch_indices_for_preds.numel():
                target_classes[batch_indices_for_preds, matched_pred_indices_batch] = target_classes_o

        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes_l1(self, 
                      pred_boxes: torch.Tensor,
                      gt_boxes_b: torch.Tensor,
                      gt_labels_b: torch.Tensor,
                      indices: List[Tuple[torch.Tensor, torch.Tensor]],
                      num_total_boxes: int
                     ) -> Dict[str, torch.Tensor]:
        batch_indices_for_preds, matched_pred_indices_batch = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[batch_indices_for_preds, matched_pred_indices_batch]

        target_boxes_list = []
        for i, (_, gt_idx_sample) in enumerate(indices):
            valid_gt_mask_sample_i = (gt_labels_b[i] >= 0)
            valid_gt_boxes_sample_i = gt_boxes_b[i][valid_gt_mask_sample_i]
            if gt_idx_sample.numel() > 0 and valid_gt_boxes_sample_i.numel() > 0:
                if gt_idx_sample.max() < valid_gt_boxes_sample_i.shape[0]:
                    target_boxes_list.append(valid_gt_boxes_sample_i[gt_idx_sample])
        
        if not target_boxes_list :
            return {'loss_bbox_l1': torch.tensor(0.0, device=pred_boxes.device)}

        target_boxes = torch.cat(target_boxes_list, dim=0) 

        if src_boxes.shape[0] != target_boxes.shape[0]:
            return {'loss_bbox_l1': torch.tensor(0.0, device=pred_boxes.device)}

        loss_bbox_l1 = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox_l1'] = loss_bbox_l1.sum() / num_total_boxes
        return losses

    def loss_boxes_giou(self, 
                        pred_boxes: torch.Tensor,
                        gt_boxes_b: torch.Tensor,
                        gt_labels_b: torch.Tensor,
                        indices: List[Tuple[torch.Tensor, torch.Tensor]],
                        num_total_boxes: int
                       ) -> Dict[str, torch.Tensor]:
        batch_indices_for_preds, matched_pred_indices_batch = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[batch_indices_for_preds, matched_pred_indices_batch] # (NumMatched, 7)

        target_boxes_list = []
        for i, (_, gt_idx_sample) in enumerate(indices):
            valid_gt_mask_sample_i = (gt_labels_b[i] >= 0)
            valid_gt_boxes_sample_i = gt_boxes_b[i][valid_gt_mask_sample_i]
            if gt_idx_sample.numel() > 0 and valid_gt_boxes_sample_i.numel() > 0:
                if gt_idx_sample.max() < valid_gt_boxes_sample_i.shape[0]:
                     target_boxes_list.append(valid_gt_boxes_sample_i[gt_idx_sample])
        
        if not target_boxes_list or src_boxes.numel() == 0: # Prüfe auch src_boxes
            return {'loss_giou': torch.tensor(0.0, device=pred_boxes.device)}
            
        target_boxes = torch.cat(target_boxes_list, dim=0) # (NumMatched, 7)
        
        if src_boxes.shape[0] != target_boxes.shape[0]:
            return {'loss_giou': torch.tensor(0.0, device=pred_boxes.device)}

        # Berechne paarweise GIoU-Werte für die gematchten Paare
        # generalized_bev_iou_pytorch erwartet (N,7) und (M,7) und gibt (N,M) zurück.
        # Hier haben wir bereits die 1-zu-1 gematchten Paare.
        giou_values_per_pair = []
        for k_pair in range(src_boxes.shape[0]):
            # Unsqueeze, um (1,7) zu erhalten, da die Funktion Batch-Verarbeitung erwartet
            giou_val = generalized_bev_iou_pytorch(src_boxes[k_pair].unsqueeze(0), 
                                                   target_boxes[k_pair].unsqueeze(0))
            giou_values_per_pair.append(giou_val.squeeze()) # Squeeze, um Skalar zu erhalten
        
        losses = {}
        if giou_values_per_pair:
            giou_values_tensor = torch.stack(giou_values_per_pair) # (NumMatched)
            loss_giou = (1.0 - giou_values_tensor).sum() / num_total_boxes # GIoU-Loss ist 1 - GIoU
            losses['loss_giou'] = loss_giou
        else:
            losses['loss_giou'] = torch.tensor(0.0, device=pred_boxes.device)
        return losses

    def forward(self, 
                decoder_outputs: Dict[str, torch.Tensor], 
                gt_labels_b: torch.Tensor,
                gt_boxes_b: torch.Tensor
               ) -> Dict[str, torch.Tensor]:
        pred_logits = decoder_outputs['pred_logits']
        pred_boxes = decoder_outputs['pred_boxes']

        indices = self.matcher(pred_logits, pred_boxes, gt_labels_b, gt_boxes_b)

        num_total_matched_boxes = sum(len(t[0]) for t in indices)
        num_total_matched_boxes = torch.as_tensor([num_total_matched_boxes], dtype=torch.float, device=pred_logits.device)
        num_total_matched_boxes = torch.clamp(num_total_matched_boxes, min=1).item()

        losses = {}
        for loss_type in self.losses:
            if loss_type == 'labels':
                losses.update(self.loss_labels(pred_logits, gt_labels_b, indices, num_total_matched_boxes))
            elif loss_type == 'boxes_l1':
                losses.update(self.loss_boxes_l1(pred_boxes, gt_boxes_b, gt_labels_b, indices, num_total_matched_boxes))
            elif loss_type == 'giou_bev': # Name aus der Config 'losses_to_compute'
                # Intern wird der Schlüssel 'loss_giou' verwendet, was mit 'loss_weight_dict' übereinstimmt.
                losses.update(self.loss_boxes_giou(pred_boxes, gt_boxes_b, gt_labels_b, indices, num_total_matched_boxes))
            else:
                raise ValueError(f"Unbekannter Verlusttyp in self.losses: {loss_type}")
        return losses

if __name__ == '__main__':
    print("Running SetCriterion and HungarianMatcher example with config-loaded parameters...")

    config_file_path = "config/pipeline_c_modules.yaml"
    if not os.path.exists(config_file_path):
        alt_config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'pipeline_c_modules.yaml')
        if os.path.exists(alt_config_path):
            config_file_path = alt_config_path
        else:
            print(f"ERROR: Config file not found at {config_file_path} or {alt_config_path}")
            # Fallback-Konfiguration, wenn keine Datei gefunden wird
            cfg_dict = {
                "model": {"num_classes": 5, "num_queries": 10, "box_dim": 7},
                "loss": {
                    "cost_class_weight": 1.0, "cost_bbox_l1_weight": 1.0, "cost_giou_bev_weight": 1.0,
                    "eos_coefficient": 0.1, "losses_to_compute": ["labels", "boxes_l1", "giou_bev"],
                    "loss_weight_dict": {"loss_ce": 1.0, "loss_bbox_l1": 1.0, "loss_giou": 1.0}
                }
            }
            full_pipeline_config = OmegaConf.create(cfg_dict)
            print("Using fallback configuration for loss test.")
            # exit() # Nicht beenden, sondern mit Fallback fortfahren
    
    if os.path.exists(config_file_path): # Nur laden, wenn Pfad existiert
        try:
            cfg_dict = load_config(config_file_path)
            full_pipeline_config = OmegaConf.create(cfg_dict) 
            print(f"Konfiguration für Loss-Test geladen von: {os.path.abspath(config_file_path)}")
        except Exception as e:
            print(f"Fehler beim Laden der Konfiguration für Loss-Test: {e}")
            cfg_dict = {
                "model": {"num_classes": 5, "num_queries": 10, "box_dim": 7},
                "loss": {
                    "cost_class_weight": 1.0, "cost_bbox_l1_weight": 1.0, "cost_giou_bev_weight": 1.0,
                    "eos_coefficient": 0.1, "losses_to_compute": ["labels", "boxes_l1", "giou_bev"],
                    "loss_weight_dict": {"loss_ce": 1.0, "loss_bbox_l1": 1.0, "loss_giou": 1.0}
                }
            }
            full_pipeline_config = OmegaConf.create(cfg_dict)
            print(f"Verwende Standard-Fallback-Parameter aufgrund eines Ladefehlers: {cfg_dict}")
    
    model_cfg = full_pipeline_config.get('model', OmegaConf.create({})) 
    loss_cfg = full_pipeline_config.get('loss', OmegaConf.create({}))
    
    num_classes_cfg = model_cfg.get('num_classes', 5) 
    num_queries_cfg = model_cfg.get('num_queries', 10)
    box_dim_cfg = model_cfg.get('box_dim', 7)
    cost_class_cfg = loss_cfg.get('cost_class_weight', 1.0)
    cost_bbox_l1_cfg = loss_cfg.get('cost_bbox_l1_weight', 1.0)
    cost_giou_bev_cfg = loss_cfg.get('cost_giou_bev_weight', 1.0)
    eos_coef_cfg = loss_cfg.get('eos_coefficient', 0.1)
    losses_to_compute_cfg = list(OmegaConf.to_container(loss_cfg.get('losses_to_compute', ['labels', 'boxes_l1', 'giou_bev']), resolve=True))
    loss_weight_dict_cfg = dict(OmegaConf.to_container(loss_cfg.get('loss_weight_dict', {'loss_ce': 1.0, 'loss_bbox_l1': 1.0, 'loss_giou': 1.0}), resolve=True))

    batch_size = 2
    device = torch.device("cpu") # Test auf CPU
    dummy_pred_logits = torch.rand(batch_size, num_queries_cfg, num_classes_cfg + 1, device=device) 
    # Box-Parameter: cx, cy, cz, w, l, h, yaw
    # cx, cy im Bereich [-50, 50], cz ~0, w,l,h ~[1,5], yaw [-pi, pi]
    dummy_pred_boxes = torch.rand(batch_size, num_queries_cfg, box_dim_cfg, device=device)
    dummy_pred_boxes[:, :, 0:2] = (dummy_pred_boxes[:, :, 0:2] * 100) - 50 # cx, cy
    dummy_pred_boxes[:, :, 2] = dummy_pred_boxes[:, :, 2] * 2 - 1       # cz
    dummy_pred_boxes[:, :, 3:6] = (dummy_pred_boxes[:, :, 3:6] * 4) + 1 # w,l,h
    dummy_pred_boxes[:, :, 6] = (dummy_pred_boxes[:, :, 6] * 2 * math.pi) - math.pi # yaw

    max_gt_objs_in_batch_for_test = 4 
    dummy_gt_labels_b = torch.full((batch_size, max_gt_objs_in_batch_for_test), -1, dtype=torch.long, device=device) 
    dummy_gt_boxes_b = torch.zeros((batch_size, max_gt_objs_in_batch_for_test, box_dim_cfg), dtype=torch.float32, device=device)

    if max_gt_objs_in_batch_for_test >= 3:
        dummy_gt_labels_b[0, :3] = torch.randint(0, num_classes_cfg, (3,), device=device)
        gt_boxes_sample0 = torch.rand(3, box_dim_cfg, device=device)
        gt_boxes_sample0[:, 0:2] = (gt_boxes_sample0[:, 0:2] * 80) - 40
        gt_boxes_sample0[:, 3:6] = (gt_boxes_sample0[:, 3:6] * 3) + 1
        gt_boxes_sample0[:, 6] = (gt_boxes_sample0[:, 6] * 2 * math.pi) - math.pi
        dummy_gt_boxes_b[0, :3, :] = gt_boxes_sample0
        
    if max_gt_objs_in_batch_for_test >= 2:
        dummy_gt_labels_b[1, :2] = torch.randint(0, num_classes_cfg, (2,), device=device)
        gt_boxes_sample1 = torch.rand(2, box_dim_cfg, device=device)
        gt_boxes_sample1[:, 0:2] = (gt_boxes_sample1[:, 0:2] * 70) - 35
        gt_boxes_sample1[:, 3:6] = (gt_boxes_sample1[:, 3:6] * 4) + 0.5
        gt_boxes_sample1[:, 6] = (gt_boxes_sample1[:, 6] * 2 * math.pi) - math.pi
        dummy_gt_boxes_b[1, :2, :] = gt_boxes_sample1
    
    print(f"\nVerwendete Loss-Parameter (aus Config oder Fallback):")
    print(f"  num_classes (ohne BG): {num_classes_cfg}, num_queries: {num_queries_cfg}, box_dim: {box_dim_cfg}")
    print(f"  Matcher Kosten: class={cost_class_cfg}, bbox_l1={cost_bbox_l1_cfg}, giou_bev={cost_giou_bev_cfg}")
    print(f"  Criterion: eos_coef={eos_coef_cfg}, losses={losses_to_compute_cfg}, weights={loss_weight_dict_cfg}")

    matcher_instance = HungarianMatcher(
        cost_class=cost_class_cfg, 
        cost_bbox_l1=cost_bbox_l1_cfg, 
        cost_giou_bev=cost_giou_bev_cfg
    )
    criterion = SetCriterion(
        num_classes=num_classes_cfg, 
        matcher=matcher_instance, 
        eos_coef=eos_coef_cfg,
        losses=losses_to_compute_cfg,
        weight_dict=loss_weight_dict_cfg
    ).to(device)
    
    print("\nTeste HungarianMatcher separat:")
    matched_indices = matcher_instance(dummy_pred_logits, dummy_pred_boxes, dummy_gt_labels_b, dummy_gt_boxes_b)
    for i_sample, (pred_idx, gt_idx) in enumerate(matched_indices):
        print(f"  Sample {i_sample}: Matched Pred Indices: {pred_idx.tolist()}, Matched GT Indices (relativ zu validen GTs): {gt_idx.tolist()}")

    print("\nTeste SetCriterion.forward():")
    calculated_losses = criterion(
        decoder_outputs={"pred_logits": dummy_pred_logits, "pred_boxes": dummy_pred_boxes},
        gt_labels_b=dummy_gt_labels_b, 
        gt_boxes_b=dummy_gt_boxes_b    
    )

    print("\nBerechnete Verluste (ungewichtet):")
    for loss_name, loss_value in calculated_losses.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")
    
    total_weighted_loss = torch.tensor(0.0, device=device)
    if calculated_losses: 
        for k_loss in calculated_losses.keys():
            if k_loss in loss_weight_dict_cfg: 
                total_weighted_loss += calculated_losses[k_loss] * loss_weight_dict_cfg[k_loss]
    print(f"  Gewichteter Gesamtverlust: {total_weighted_loss.item():.4f}")
    
    print("\nSetCriterion and HungarianMatcher example run successful.")
