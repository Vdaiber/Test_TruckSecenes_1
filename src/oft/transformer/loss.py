# src/oft/transformer/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional, Tuple
import numpy as np
import math # Für math.pi

import os
# Importiere load_config nur, wenn es im Test-Block benötigt wird
# from oft.utils.config import load_config 
from omegaconf import OmegaConf, DictConfig # DictConfig hinzugefügt

# --- Hilfsfunktionen für BEV GIoU (unverändert) ---

def get_bev_corners_pytorch(boxes_7d: torch.Tensor) -> torch.Tensor:
    """
    Berechnet die 4 BEV-Eckpunkte für einen Satz von 7D-Boxen.
    Args:
        boxes_7d: Tensor der Form (N, 7) mit [cx, cy, cz, width, length, height, yaw].
                  width (dim 3) ist entlang der lokalen y-Achse der Box.
                  length (dim 4) ist entlang der lokalen x-Achse der Box.
    Returns:
        Tensor der Form (N, 4, 2) mit den Eckpunkten (x,y) in der BEV-Ebene.
    """
    if boxes_7d.ndim == 1:
        boxes_7d = boxes_7d.unsqueeze(0)
    
    centers_x = boxes_7d[:, 0]
    centers_y = boxes_7d[:, 1]
    widths = boxes_7d[:, 3] 
    lengths = boxes_7d[:, 4]
    yaws = boxes_7d[:, 6]

    half_lengths = lengths / 2.0
    half_widths = widths / 2.0

    corners_local = torch.zeros((boxes_7d.shape[0], 4, 2), device=boxes_7d.device, dtype=boxes_7d.dtype) 
    corners_local[:, 0, 0] = half_lengths
    corners_local[:, 0, 1] = half_widths
    corners_local[:, 1, 0] = half_lengths
    corners_local[:, 1, 1] = -half_widths
    corners_local[:, 2, 0] = -half_lengths
    corners_local[:, 2, 1] = -half_widths
    corners_local[:, 3, 0] = -half_lengths
    corners_local[:, 3, 1] = half_widths

    cos_yaw = torch.cos(yaws)
    sin_yaw = torch.sin(yaws)

    corners_rotated_x = corners_local[..., 0] * cos_yaw.unsqueeze(1) - corners_local[..., 1] * sin_yaw.unsqueeze(1)
    corners_rotated_y = corners_local[..., 0] * sin_yaw.unsqueeze(1) + corners_local[..., 1] * cos_yaw.unsqueeze(1)
    
    corners_rotated = torch.stack((corners_rotated_x, corners_rotated_y), dim=-1)
    corners_world = corners_rotated + torch.stack((centers_x, centers_y), dim=-1).unsqueeze(1)
    
    return corners_world

def calculate_bev_iou_from_corners_pytorch(
    corners1: torch.Tensor, 
    corners2: torch.Tensor, 
    areas1: torch.Tensor,   
    areas2: torch.Tensor     
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = corners1.shape[0]
    M = corners2.shape[0]
    iou_matrix = torch.zeros((N, M), device=corners1.device, dtype=corners1.dtype) 
    intersection_area_matrix = torch.zeros((N, M), device=corners1.device, dtype=corners1.dtype)

    min_xy1, _ = torch.min(corners1, dim=1) 
    max_xy1, _ = torch.max(corners1, dim=1) 
    
    min_xy2, _ = torch.min(corners2, dim=1) 
    max_xy2, _ = torch.max(corners2, dim=1) 

    for i in range(N):
        for j in range(M):
            inter_min_x = torch.max(min_xy1[i, 0], min_xy2[j, 0])
            inter_min_y = torch.max(min_xy1[i, 1], min_xy2[j, 1])
            inter_max_x = torch.min(max_xy1[i, 0], max_xy2[j, 0])
            inter_max_y = torch.min(max_xy1[i, 1], max_xy2[j, 1])

            inter_width = torch.clamp(inter_max_x - inter_min_x, min=0.0)
            inter_height = torch.clamp(inter_max_y - inter_min_y, min=0.0)
            intersection_aabb_corners = inter_width * inter_height
            intersection_area_matrix[i, j] = intersection_aabb_corners

            union_area = areas1[i] + areas2[j] - intersection_aabb_corners
            if union_area > 1e-7: 
                iou_matrix[i, j] = intersection_aabb_corners / union_area
            else:
                iou_matrix[i, j] = torch.tensor(0.0, device=corners1.device, dtype=corners1.dtype)
    
    return torch.clamp(iou_matrix, min=0.0, max=1.0), intersection_area_matrix


def calculate_enclosing_box_area_bev_pytorch(
    corners1: torch.Tensor, 
    corners2: torch.Tensor  
) -> torch.Tensor:
    N = corners1.shape[0]
    M = corners2.shape[0]
    enclosing_area_matrix = torch.zeros((N, M), device=corners1.device, dtype=corners1.dtype) 

    for i in range(N):
        for j in range(M):
            all_corners = torch.cat((corners1[i], corners2[j]), dim=0) 
            
            min_coords, _ = torch.min(all_corners, dim=0) 
            max_coords, _ = torch.max(all_corners, dim=0) 
            
            enclosing_width = max_coords[0] - min_coords[0]
            enclosing_height = max_coords[1] - min_coords[1]
            enclosing_area_matrix[i, j] = enclosing_width * enclosing_height
            
    return enclosing_area_matrix

def generalized_bev_iou_pytorch(boxes1_7d: torch.Tensor, boxes2_7d: torch.Tensor) -> torch.Tensor:
    """ Berechnet GIoU für 7D Boxen im BEV. Erwartet Boxen mit TATSÄCHLICHEN Dimensionen. """
    eps = 1e-7 

    # Dimensionen sind an Index 3 (width) und 4 (length)
    areas1 = boxes1_7d[:, 3] * boxes1_7d[:, 4] 
    areas2 = boxes2_7d[:, 3] * boxes2_7d[:, 4] 

    corners1 = get_bev_corners_pytorch(boxes1_7d) 
    corners2 = get_bev_corners_pytorch(boxes2_7d) 

    iou_approx, intersection_area_approx = calculate_bev_iou_from_corners_pytorch(corners1, corners2, areas1, areas2)

    enclosing_area = calculate_enclosing_box_area_bev_pytorch(corners1, corners2) 

    union_area = areas1.unsqueeze(1) + areas2.unsqueeze(0) - intersection_area_approx 
    union_area = torch.clamp(union_area, min=eps)

    giou = iou_approx - (enclosing_area - union_area) / (enclosing_area + eps)
    
    return torch.clamp(giou, min=-1.0, max=1.0)


class HungarianMatcher(nn.Module):
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
                pred_logits: torch.Tensor,
                pred_boxes_for_loss: torch.Tensor, # Mit log-dims, für L1-Kosten im Matcher
                pred_boxes_for_matching: torch.Tensor, # Mit aktuellen dims, für GIoU-Kosten im Matcher
                gt_labels_b: torch.Tensor,
                gt_boxes_b_log_dims: torch.Tensor, # Mit log-dims
                gt_boxes_b_actual_dims: torch.Tensor # Mit aktuellen dims
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
            current_gt_boxes_log_dims_i = gt_boxes_b_log_dims[i][valid_gt_mask_i]
            current_gt_boxes_actual_dims_i = gt_boxes_b_actual_dims[i][valid_gt_mask_i]
            num_gt_i = current_gt_labels_i.shape[0]

            if num_gt_i == 0:
                indices.append((torch.tensor([], dtype=torch.long, device=pred_logits.device),
                                torch.tensor([], dtype=torch.long, device=pred_logits.device)))
                continue

            current_pred_logits_i = pred_logits[i]
            current_pred_boxes_for_loss_i = pred_boxes_for_loss[i]
            current_pred_boxes_for_matching_i = pred_boxes_for_matching[i]

            prob = current_pred_logits_i.softmax(-1)
            cost_class_matrix = -prob[:, current_gt_labels_i]

            cost_bbox_l1_matrix = torch.cdist(current_pred_boxes_for_loss_i, current_gt_boxes_log_dims_i, p=1)
            
            giou_values = generalized_bev_iou_pytorch(current_pred_boxes_for_matching_i, current_gt_boxes_actual_dims_i)
            cost_giou_matrix = 1.0 - giou_values 

            C = (self.cost_class * cost_class_matrix +
                 self.cost_bbox_l1 * cost_bbox_l1_matrix +
                 self.cost_giou_bev * cost_giou_matrix)
            
            C_np = C.detach().cpu().numpy()
            if C_np.size == 0:
                 indices.append((torch.tensor([], dtype=torch.long, device=pred_logits.device),
                                torch.tensor([], dtype=torch.long, device=pred_logits.device)))
                 continue
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
                 weight_dict: Dict[str, float],
                 coord_normalization_factor: float = 1.0 
                ): 
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef 
        self.losses = losses
        self.weight_dict = weight_dict
        self._print_count_l1_loss = 0 
        self.coord_normalization_factor = coord_normalization_factor

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
            try:
                target_classes_o = torch.cat(target_classes_o_list)
                if target_classes_o.numel() == batch_indices_for_preds.numel():
                    target_classes[batch_indices_for_preds, matched_pred_indices_batch] = target_classes_o
            except RuntimeError as e:
                pass

        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes_l1(self, 
                      pred_boxes_for_loss: torch.Tensor,    # (B, NumQueries, 7) -> (cx,cy,cz, log(w),log(l),log(h), yaw)
                      gt_boxes_b_log_dims: torch.Tensor,    # (B, MaxGTObjects, 7) -> mit log(w,l,h)
                      gt_labels_b: torch.Tensor, 
                      indices: List[Tuple[torch.Tensor, torch.Tensor]],
                      num_total_boxes: int
                     ) -> Dict[str, torch.Tensor]:
        batch_indices_for_preds, matched_pred_indices_batch = self._get_src_permutation_idx(indices)
        src_boxes_log_dims = pred_boxes_for_loss[batch_indices_for_preds, matched_pred_indices_batch]

        target_boxes_log_dims_list = []
        for i, (_, gt_idx_sample) in enumerate(indices):
            valid_gt_mask_sample_i = (gt_labels_b[i] >= 0)
            valid_gt_boxes_log_dims_sample_i = gt_boxes_b_log_dims[i][valid_gt_mask_sample_i]
            if gt_idx_sample.numel() > 0 and valid_gt_boxes_log_dims_sample_i.numel() > 0:
                if gt_idx_sample.max() < valid_gt_boxes_log_dims_sample_i.shape[0]:
                    target_boxes_log_dims_list.append(valid_gt_boxes_log_dims_sample_i[gt_idx_sample])
        
        if not target_boxes_log_dims_list or src_boxes_log_dims.numel() == 0: 
            return {'loss_bbox_l1': torch.tensor(0.0, device=pred_boxes_for_loss.device, dtype=pred_boxes_for_loss.dtype)} 

        target_boxes_log_dims = torch.cat(target_boxes_log_dims_list, dim=0) 

        if src_boxes_log_dims.shape[0] != target_boxes_log_dims.shape[0]:
            return {'loss_bbox_l1': torch.tensor(0.0, device=pred_boxes_for_loss.device, dtype=pred_boxes_for_loss.dtype)} 

        src_centers_norm = src_boxes_log_dims[..., :3] / self.coord_normalization_factor
        tgt_centers_norm = target_boxes_log_dims[..., :3] / self.coord_normalization_factor
        loss_centers_l1 = F.l1_loss(src_centers_norm, tgt_centers_norm, reduction='none')
        
        loss_log_dims_l1 = F.l1_loss(src_boxes_log_dims[..., 3:6], target_boxes_log_dims[..., 3:6], reduction='none')
        
        loss_yaw_l1 = F.l1_loss(src_boxes_log_dims[..., 6:7], target_boxes_log_dims[..., 6:7], reduction='none')
        
        loss_bbox_l1_all_params = torch.cat((loss_centers_l1, loss_log_dims_l1, loss_yaw_l1), dim=-1)
        
        if self._print_count_l1_loss < 1: 
            print(f"\n--- DEBUG loss_boxes_l1 (Call {self._print_count_l1_loss + 1}) ---")
            print(f"  coord_normalization_factor (für Zentren): {self.coord_normalization_factor}")
            print(f"  Anzahl gematchter Boxen im Batch: {src_boxes_log_dims.shape[0]}")
            if src_boxes_log_dims.shape[0] > 0:
                num_to_print = min(src_boxes_log_dims.shape[0], 2) 
                for k_debug in range(num_to_print):
                    print(f"    Paar {k_debug+1}:")
                    print(f"      Pred Box (log-dims): {np.round(src_boxes_log_dims[k_debug].detach().cpu().numpy(), 2)}")
                    print(f"      GT Box   (log-dims): {np.round(target_boxes_log_dims[k_debug].detach().cpu().numpy(), 2)}")
                    print(f"      Pred Centers NORM: {np.round(src_centers_norm[k_debug].detach().cpu().numpy(), 4)}")
                    print(f"      GT Centers   NORM: {np.round(tgt_centers_norm[k_debug].detach().cpu().numpy(), 4)}")
            self._print_count_l1_loss += 1

        losses = {}
        losses['loss_bbox_l1'] = loss_bbox_l1_all_params.sum() / num_total_boxes 
        return losses

    def loss_boxes_giou(self, 
                        pred_boxes_for_matching: torch.Tensor, 
                        gt_boxes_b_actual_dims: torch.Tensor,  
                        gt_labels_b: torch.Tensor, 
                        indices: List[Tuple[torch.Tensor, torch.Tensor]],
                        num_total_boxes: int
                       ) -> Dict[str, torch.Tensor]:
        batch_indices_for_preds, matched_pred_indices_batch = self._get_src_permutation_idx(indices)
        src_boxes_actual_dims = pred_boxes_for_matching[batch_indices_for_preds, matched_pred_indices_batch] 

        target_boxes_actual_dims_list = []
        for i, (_, gt_idx_sample) in enumerate(indices):
            valid_gt_mask_sample_i = (gt_labels_b[i] >= 0)
            valid_gt_boxes_actual_dims_sample_i = gt_boxes_b_actual_dims[i][valid_gt_mask_sample_i]
            if gt_idx_sample.numel() > 0 and valid_gt_boxes_actual_dims_sample_i.numel() > 0:
                if gt_idx_sample.max() < valid_gt_boxes_actual_dims_sample_i.shape[0]:
                      target_boxes_actual_dims_list.append(valid_gt_boxes_actual_dims_sample_i[gt_idx_sample])
        
        if not target_boxes_actual_dims_list or src_boxes_actual_dims.numel() == 0: 
            return {'loss_giou': torch.tensor(0.0, device=pred_boxes_for_matching.device, dtype=pred_boxes_for_matching.dtype)} 
             
        target_boxes_actual_dims = torch.cat(target_boxes_actual_dims_list, dim=0)
         
        if src_boxes_actual_dims.shape[0] != target_boxes_actual_dims.shape[0]:
            return {'loss_giou': torch.tensor(0.0, device=pred_boxes_for_matching.device, dtype=pred_boxes_for_matching.dtype)} 

        if src_boxes_actual_dims.shape[0] > 0 :
            giou_matrix_for_matched_pairs = generalized_bev_iou_pytorch(src_boxes_actual_dims, target_boxes_actual_dims)
            giou_values_tensor = torch.diag(giou_matrix_for_matched_pairs)
            loss_giou = (1.0 - giou_values_tensor).sum() / num_total_boxes 
        else:
            loss_giou = torch.tensor(0.0, device=pred_boxes_for_matching.device, dtype=pred_boxes_for_matching.dtype)

        losses = {}
        losses['loss_giou'] = loss_giou
        return losses

    def forward(self, 
                decoder_outputs: Dict[str, torch.Tensor],
                gt_labels_b: torch.Tensor,
                gt_boxes_b_log_dims: torch.Tensor,
                gt_boxes_b_actual_dims: torch.Tensor,
                gt_valid_mask_b: torch.Tensor # Wird von Collate bereitgestellt, hier aber nicht direkt verwendet
               ) -> Dict[str, torch.Tensor]:
        
        pred_logits = decoder_outputs['pred_logits']
        pred_boxes_for_loss = decoder_outputs['pred_boxes_for_loss']
        pred_boxes_for_matching = decoder_outputs['pred_boxes_for_matching_and_giou']

        indices = self.matcher(pred_logits, 
                               pred_boxes_for_loss,       
                               pred_boxes_for_matching,   
                               gt_labels_b, 
                               gt_boxes_b_log_dims,       
                               gt_boxes_b_actual_dims)    

        num_total_matched_boxes = sum(len(t[0]) for t in indices)
        num_total_matched_boxes = torch.as_tensor([num_total_matched_boxes], dtype=torch.float, device=pred_logits.device)
        num_total_matched_boxes = torch.clamp(num_total_matched_boxes, min=1).item()

        losses = {}
        for loss_type in self.losses:
            if loss_type == 'labels':
                losses.update(self.loss_labels(pred_logits, gt_labels_b, indices, num_total_matched_boxes))
            elif loss_type == 'boxes_l1':
                losses.update(self.loss_boxes_l1(pred_boxes_for_loss, gt_boxes_b_log_dims, gt_labels_b, indices, num_total_matched_boxes))
            elif loss_type == 'giou_bev':
                losses.update(self.loss_boxes_giou(pred_boxes_for_matching, gt_boxes_b_actual_dims, gt_labels_b, indices, num_total_matched_boxes))
            else:
                raise ValueError(f"Unbekannter Verlusttyp in self.losses: {loss_type}")
        return losses

if __name__ == '__main__':
    print("Running SetCriterion and HungarianMatcher example (mit Log-Dims)...")
    from oft.utils.config import load_config 

    config_file_path_loss = "config/pipeline_c_modules.yaml"
    project_root_loss = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    config_file_path_loss_abs = os.path.join(project_root_loss, config_file_path_loss)
    
    cfg_dict_for_test_loss = None
    if not os.path.exists(config_file_path_loss_abs):
        print(f"ERROR: Config file not found at {config_file_path_loss_abs}")
        from truckscenes.eval.detection.constants import DETECTION_NAMES as TRUCKSCENES_DET_NAMES_LOSS_TEST
        cfg_dict_for_test_loss = { 
            "model": {"num_classes": 12, "num_queries": 10, "box_dim": 7, "pe_max_coord_val": 150.0},
            "loss": { "cost_class_weight": 2.0, "cost_bbox_l1_weight": 5.0, "cost_giou_bev_weight": 2.0,
                      "eos_coefficient": 0.1, "losses_to_compute": ["labels", "boxes_l1", "giou_bev"],
                      "loss_weight_dict": {"loss_ce": 1.0, "loss_bbox_l1": 5.0, "loss_giou": 2.0} }}
        print("Verwende Fallback-Konfiguration für Loss-Test.")
    else:
        cfg_dict_for_test_loss = load_config(config_file_path_loss_abs)
        if not isinstance(cfg_dict_for_test_loss, dict):
            cfg_dict_for_test_loss = OmegaConf.to_container(cfg_dict_for_test_loss, resolve=True)
        print(f"Konfiguration für Loss-Test geladen von: {config_file_path_loss_abs}")

    model_cfg_loss = cfg_dict_for_test_loss.get('model', {})
    loss_cfg_loss = cfg_dict_for_test_loss.get('loss', {})
     
    num_classes_cfg = model_cfg_loss.get('num_classes', 12)
    num_queries_cfg = model_cfg_loss.get('num_queries', 100)
    box_dim_cfg = model_cfg_loss.get('box_dim', 7)
    
    cost_class_cfg = loss_cfg_loss.get('cost_class_weight', 2.0)
    cost_bbox_l1_cfg = loss_cfg_loss.get('cost_bbox_l1_weight', 5.0)
    cost_giou_bev_cfg = loss_cfg_loss.get('cost_giou_bev_weight', 2.0)
    eos_coef_cfg = loss_cfg_loss.get('eos_coefficient', 0.1)
    losses_to_compute_cfg = loss_cfg_loss.get('losses_to_compute', ["labels", "boxes_l1", "giou_bev"])
    loss_weight_dict_cfg = loss_cfg_loss.get('loss_weight_dict', {"loss_ce": 1.0, "loss_bbox_l1": 5.0, "loss_giou": 2.0})
    coord_norm_factor_cfg = model_cfg_loss.get('pe_max_coord_val', 150.0)


    batch_s = 2
    device = torch.device("cpu") 
    dummy_pred_logits = torch.rand(batch_s, num_queries_cfg, num_classes_cfg + 1, device=device)
    dummy_pred_boxes_for_loss = torch.randn(batch_s, num_queries_cfg, box_dim_cfg, device=device) 
    dummy_pred_boxes_for_loss[..., 3:6] = torch.rand(batch_s, num_queries_cfg, 3, device=device) * 2 - 1 
    dummy_pred_boxes_for_matching = torch.randn(batch_s, num_queries_cfg, box_dim_cfg, device=device)
    dummy_pred_boxes_for_matching[..., :3] = dummy_pred_boxes_for_loss[..., :3].clone().detach()
    dummy_pred_boxes_for_matching[..., 3:6] = torch.exp(dummy_pred_boxes_for_loss[..., 3:6].clone().detach())
    dummy_pred_boxes_for_matching[..., 6:7] = dummy_pred_boxes_for_loss[..., 6:7].clone().detach()


    max_gt_objs_for_test = 5
    dummy_gt_labels_b = torch.randint(0, num_classes_cfg, (batch_s, max_gt_objs_for_test), device=device)
    dummy_gt_boxes_b_log_dims = torch.randn(batch_s, max_gt_objs_for_test, box_dim_cfg, device=device)
    dummy_gt_boxes_b_log_dims[..., 3:6] = torch.rand(batch_s, max_gt_objs_for_test, 3, device=device) * 2 - 0.5
    dummy_gt_boxes_b_actual_dims = dummy_gt_boxes_b_log_dims.clone()
    dummy_gt_boxes_b_actual_dims[..., 3:6] = torch.exp(dummy_gt_boxes_b_log_dims[..., 3:6])
    
    dummy_gt_valid_mask_b = torch.ones((batch_s, max_gt_objs_for_test), dtype=torch.bool, device=device)
    if max_gt_objs_for_test > 2:
        dummy_gt_labels_b[0, -2:] = -1 
        dummy_gt_valid_mask_b[0, -2:] = False


    print(f"\nVerwendete Loss-Parameter:")
    print(f"  Matcher Kosten: class={cost_class_cfg}, bbox_l1={cost_bbox_l1_cfg}, giou_bev={cost_giou_bev_cfg}")
    print(f"  Criterion: eos_coef={eos_coef_cfg}, coord_norm_factor (Zentren)={coord_norm_factor_cfg}")

    matcher_instance = HungarianMatcher(
        cost_class=cost_class_cfg, cost_bbox_l1=cost_bbox_l1_cfg, cost_giou_bev=cost_giou_bev_cfg
    )
    criterion = SetCriterion(
        num_classes=num_classes_cfg, matcher=matcher_instance, eos_coef=eos_coef_cfg,
        losses=losses_to_compute_cfg, weight_dict=loss_weight_dict_cfg,
        coord_normalization_factor=coord_norm_factor_cfg
    ).to(device)
    criterion._print_count_l1_loss = 0 

    print("\nTeste SetCriterion.forward():")
    decoder_outputs_test = {
        "pred_logits": dummy_pred_logits,
        "pred_boxes_for_loss": dummy_pred_boxes_for_loss,
        "pred_boxes_for_matching_and_giou": dummy_pred_boxes_for_matching
    }
    calculated_losses = criterion(
        decoder_outputs=decoder_outputs_test,
        gt_labels_b=dummy_gt_labels_b,
        gt_boxes_b_log_dims=dummy_gt_boxes_b_log_dims,
        gt_boxes_b_actual_dims=dummy_gt_boxes_b_actual_dims,
        gt_valid_mask_b=dummy_gt_valid_mask_b
    )

    print("\nBerechnete Verluste (ungewichtet):")
    for loss_name, loss_value in calculated_losses.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")
     
    print("\nSetCriterion and HungarianMatcher example run successful (mit Log-Dims).")