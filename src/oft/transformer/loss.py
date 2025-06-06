# src/oft/transformer/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional, Tuple
import numpy as np
import math

import os
# from oft.utils.config import load_config # Nur für __main__
from omegaconf import OmegaConf, DictConfig # Nur für __main__

# --- Hilfsfunktionen für BEV GIoU (BLEIBEN WIE VORHER) ---
def get_bev_corners_pytorch(boxes_7d: torch.Tensor) -> torch.Tensor:
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
    corners_local[:, 0, 0] = half_lengths; corners_local[:, 0, 1] = half_widths
    corners_local[:, 1, 0] = half_lengths; corners_local[:, 1, 1] = -half_widths
    corners_local[:, 2, 0] = -half_lengths; corners_local[:, 2, 1] = -half_widths
    corners_local[:, 3, 0] = -half_lengths; corners_local[:, 3, 1] = half_widths
    cos_yaw = torch.cos(yaws); sin_yaw = torch.sin(yaws)
    corners_rotated_x = corners_local[..., 0] * cos_yaw.unsqueeze(1) - corners_local[..., 1] * sin_yaw.unsqueeze(1)
    corners_rotated_y = corners_local[..., 0] * sin_yaw.unsqueeze(1) + corners_local[..., 1] * cos_yaw.unsqueeze(1)
    corners_rotated = torch.stack((corners_rotated_x, corners_rotated_y), dim=-1)
    corners_world = corners_rotated + torch.stack((centers_x, centers_y), dim=-1).unsqueeze(1)
    return corners_world

def calculate_bev_iou_from_corners_pytorch(
    corners1: torch.Tensor, corners2: torch.Tensor, 
    areas1: torch.Tensor, areas2: torch.Tensor     
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = corners1.shape[0]; M = corners2.shape[0]
    iou_matrix = torch.zeros((N, M), device=corners1.device, dtype=corners1.dtype) 
    intersection_area_matrix = torch.zeros((N, M), device=corners1.device, dtype=corners1.dtype)
    min_xy1, _ = torch.min(corners1, dim=1); max_xy1, _ = torch.max(corners1, dim=1) 
    min_xy2, _ = torch.min(corners2, dim=1); max_xy2, _ = torch.max(corners2, dim=1) 
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
            iou_matrix[i, j] = intersection_aabb_corners / union_area if union_area > 1e-7 else torch.tensor(0.0, device=corners1.device)
    return torch.clamp(iou_matrix, min=0.0, max=1.0), intersection_area_matrix

def calculate_enclosing_box_area_bev_pytorch(corners1: torch.Tensor, corners2: torch.Tensor) -> torch.Tensor:
    N = corners1.shape[0]; M = corners2.shape[0]
    enclosing_area_matrix = torch.zeros((N, M), device=corners1.device, dtype=corners1.dtype) 
    for i in range(N):
        for j in range(M):
            all_corners = torch.cat((corners1[i], corners2[j]), dim=0) 
            min_coords, _ = torch.min(all_corners, dim=0); max_coords, _ = torch.max(all_corners, dim=0) 
            enclosing_area_matrix[i, j] = (max_coords[0] - min_coords[0]) * (max_coords[1] - min_coords[1])
    return enclosing_area_matrix

def generalized_bev_iou_pytorch(boxes1_7d_actual_dims: torch.Tensor, boxes2_7d_actual_dims: torch.Tensor) -> torch.Tensor:
    eps = 1e-7 
    areas1 = boxes1_7d_actual_dims[:, 3] * boxes1_7d_actual_dims[:, 4] 
    areas2 = boxes2_7d_actual_dims[:, 3] * boxes2_7d_actual_dims[:, 4] 
    corners1 = get_bev_corners_pytorch(boxes1_7d_actual_dims) 
    corners2 = get_bev_corners_pytorch(boxes2_7d_actual_dims) 
    iou_approx, intersection_area_approx = calculate_bev_iou_from_corners_pytorch(corners1, corners2, areas1, areas2)
    enclosing_area = calculate_enclosing_box_area_bev_pytorch(corners1, corners2) 
    union_area = areas1.unsqueeze(1) + areas2.unsqueeze(0) - intersection_area_approx 
    union_area = torch.clamp(union_area, min=eps)
    giou = iou_approx - (enclosing_area - union_area) / (enclosing_area + eps)
    return torch.clamp(giou, min=-1.0, max=1.0)
# --- Ende Hilfsfunktionen für BEV GIoU ---


class HungarianMatcher(nn.Module):
    def __init__(self,
                 cost_class: float = 1.0,
                 cost_bbox_l1_offset: float = 1.0, 
                 cost_giou_bev: float = 1.0,
                 center_offset_scale_for_matcher_cost: float = 5.0
                ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox_l1_offset = cost_bbox_l1_offset 
        self.cost_giou_bev = cost_giou_bev
        self.center_offset_scale_for_matcher_cost = center_offset_scale_for_matcher_cost
        if cost_class == 0 and cost_bbox_l1_offset == 0 and cost_giou_bev == 0:
            raise ValueError("Alle Kosten-Gewichte im Matcher dürfen nicht null sein.")

    @torch.no_grad()
    def forward(self,
                pred_logits: torch.Tensor,
                pred_box_offsets: torch.Tensor,
                gt_labels_b: torch.Tensor,
                gt_boxes_b_actual_dims: torch.Tensor,
                gt_boxes_b_log_dims: torch.Tensor
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
            current_gt_boxes_actual_dims_i = gt_boxes_b_actual_dims[i][valid_gt_mask_i]
            current_gt_boxes_log_dims_i = gt_boxes_b_log_dims[i][valid_gt_mask_i]
            num_gt_i = current_gt_labels_i.shape[0]

            if num_gt_i == 0:
                indices.append((torch.tensor([], dtype=torch.long, device=pred_logits.device),
                                torch.tensor([], dtype=torch.long, device=pred_logits.device)))
                continue

            current_pred_logits_i = pred_logits[i]
            current_pred_box_offsets_i = pred_box_offsets[i]

            prob = current_pred_logits_i.softmax(-1)
            cost_class_matrix = -prob[:, current_gt_labels_i]

            target_offsets_zeros_i = torch.zeros_like(current_pred_box_offsets_i[:, :current_gt_boxes_log_dims_i.shape[1]]) # Target-Offset ist 0, angepasst an num_gt_i
            
            # Ensure current_pred_box_offsets_i is broadcastable or correctly sliced for cdist
            # If current_pred_box_offsets_i is (NumQueries, Dim) and target_offsets_zeros_i is (NumGT, Dim)
            # we need to handle this carefully. cdist handles broadcasting.
            # Let's assume target_offsets_zeros_i has shape (NumGT, Dim) as intended
            target_offsets_zeros_for_cdist = torch.zeros(num_gt_i, current_pred_box_offsets_i.shape[-1], device=pred_logits.device, dtype=current_pred_box_offsets_i.dtype)


            cost_bbox_l1_offset_matrix = torch.cdist(current_pred_box_offsets_i, target_offsets_zeros_for_cdist, p=1)
            
            num_pred_queries_i = current_pred_box_offsets_i.shape[0]
            cost_giou_matrix = torch.zeros(num_pred_queries_i, num_gt_i, device=pred_logits.device)

            pred_center_offsets_scaled_i = current_pred_box_offsets_i[:, :3]
            pred_log_dim_offsets_i = current_pred_box_offsets_i[:, 3:6]
            pred_yaw_offsets_proc_i = current_pred_box_offsets_i[:, 6:7]

            for gt_idx_loop in range(num_gt_i):
                current_ref_center_actual = current_gt_boxes_actual_dims_i[gt_idx_loop, :3]
                current_ref_log_dims = current_gt_boxes_log_dims_i[gt_idx_loop, 3:6]
                current_ref_yaw_actual = current_gt_boxes_actual_dims_i[gt_idx_loop, 6]

                recon_centers = current_ref_center_actual.unsqueeze(0) + pred_center_offsets_scaled_i
                recon_log_dims = current_ref_log_dims.unsqueeze(0) + pred_log_dim_offsets_i
                recon_actual_dims = torch.exp(recon_log_dims)
                recon_yaws = current_ref_yaw_actual + pred_yaw_offsets_proc_i.squeeze(-1)
                recon_yaws = (recon_yaws + math.pi) % (2 * math.pi) - math.pi
                
                pred_boxes_reconstructed_for_giou_vs_one_gt = torch.cat(
                    (recon_centers, recon_actual_dims, recon_yaws.unsqueeze(-1)), dim=-1
                )
                
                giou_values_col = generalized_bev_iou_pytorch(
                    pred_boxes_reconstructed_for_giou_vs_one_gt, 
                    current_gt_boxes_actual_dims_i[gt_idx_loop].unsqueeze(0)
                )
                cost_giou_matrix[:, gt_idx_loop] = 1.0 - giou_values_col.squeeze(-1)
            
            C = (self.cost_class * cost_class_matrix +
                 self.cost_bbox_l1_offset * cost_bbox_l1_offset_matrix +
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
                 coord_normalization_factor: float = 1.0, 
                 center_offset_scale: float = 5.0
                ): 
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef 
        self.losses = losses
        self.weight_dict = weight_dict
        self._print_count_l1_loss = 0 
        self.coord_normalization_factor = coord_normalization_factor
        self.center_offset_scale = center_offset_scale

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

    def loss_labels(self, pred_logits: torch.Tensor, gt_labels_b: torch.Tensor,
                    indices: List[Tuple[torch.Tensor, torch.Tensor]], num_total_boxes: int
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
                 print(f"RuntimeError in loss_labels bei target_classes Zuweisung: {e}")
                 print(f"Shapes: target_classes_o={target_classes_o.shape if target_classes_o_list else 'N/A'}, batch_indices_for_preds={batch_indices_for_preds.shape}, matched_pred_indices_batch={matched_pred_indices_batch.shape}")

        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_ce': loss_ce}

    def loss_boxes_l1_offset(self, 
                             pred_box_offsets: torch.Tensor,
                             indices: List[Tuple[torch.Tensor, torch.Tensor]],
                             num_total_boxes: int
                            ) -> Dict[str, torch.Tensor]:
        batch_indices_for_preds, matched_pred_indices_batch = self._get_src_permutation_idx(indices)
        src_box_offsets = pred_box_offsets[batch_indices_for_preds, matched_pred_indices_batch]

        if src_box_offsets.numel() == 0: 
            return {'loss_bbox_l1_offset': torch.tensor(0.0, device=pred_box_offsets.device, dtype=pred_box_offsets.dtype)} 

        target_box_offsets = torch.zeros_like(src_box_offsets)
        loss_bbox_l1_all_params = F.l1_loss(src_box_offsets, target_box_offsets, reduction='none')
        
        if self._print_count_l1_loss < 2:
            print(f"\n--- DEBUG loss_boxes_l1_offset (Call {self._print_count_l1_loss + 1}) ---")
            print(f"  Anzahl gematchter Boxen im Batch: {src_box_offsets.shape[0]}")
            if src_box_offsets.shape[0] > 0:
                num_to_print = min(src_box_offsets.shape[0], 2) 
                for k_debug in range(num_to_print):
                    print(f"    Paar {k_debug+1}:")
                    print(f"      Pred Box Offsets (Δcx_s,Δcy_s,Δcz_s, Δlogw,Δlogl,Δlogh, Δyaw_p): {np.round(src_box_offsets[k_debug].detach().cpu().numpy(), 3)}")
                    print(f"      Target Box Offsets (sollten alle 0 sein): {np.round(target_box_offsets[k_debug].detach().cpu().numpy(), 3)}")
            self._print_count_l1_loss += 1
        
        loss_val = loss_bbox_l1_all_params.sum() / num_total_boxes if num_total_boxes > 0 else torch.tensor(0.0, device=pred_box_offsets.device)
        return {'loss_bbox_l1_offset': loss_val}


    def loss_boxes_giou(self, 
                        pred_box_offsets: torch.Tensor,
                        gt_boxes_b_actual_dims: torch.Tensor,
                        gt_boxes_b_log_dims: torch.Tensor,
                        gt_labels_b: torch.Tensor, 
                        indices: List[Tuple[torch.Tensor, torch.Tensor]],
                        num_total_boxes: int
                       ) -> Dict[str, torch.Tensor]:
        
        batch_indices_for_preds, matched_pred_indices_batch = self._get_src_permutation_idx(indices)
        src_box_offsets_matched = pred_box_offsets[batch_indices_for_preds, matched_pred_indices_batch]

        target_boxes_actual_dims_list = []
        target_boxes_log_dims_list = [] 

        for i, (_, gt_idx_sample) in enumerate(indices):
            valid_gt_mask_sample_i = (gt_labels_b[i] >= 0)
            valid_gt_boxes_actual_dims_sample_i = gt_boxes_b_actual_dims[i][valid_gt_mask_sample_i]
            valid_gt_boxes_log_dims_sample_i = gt_boxes_b_log_dims[i][valid_gt_mask_sample_i]

            if gt_idx_sample.numel() > 0:
                if gt_idx_sample.max() < valid_gt_boxes_actual_dims_sample_i.shape[0]:
                    target_boxes_actual_dims_list.append(valid_gt_boxes_actual_dims_sample_i[gt_idx_sample])
                    target_boxes_log_dims_list.append(valid_gt_boxes_log_dims_sample_i[gt_idx_sample])
        
        if not target_boxes_actual_dims_list or src_box_offsets_matched.numel() == 0: 
            return {'loss_giou': torch.tensor(0.0, device=pred_box_offsets.device)} 
             
        ref_boxes_actual_dims_matched = torch.cat(target_boxes_actual_dims_list, dim=0)
        ref_boxes_log_dims_matched = torch.cat(target_boxes_log_dims_list, dim=0)
         
        if src_box_offsets_matched.shape[0] != ref_boxes_actual_dims_matched.shape[0]:
            print(f"WARNUNG GIoU: Shape-Mismatch src_box_offsets ({src_box_offsets_matched.shape[0]}) vs ref_boxes_actual ({ref_boxes_actual_dims_matched.shape[0]})")
            return {'loss_giou': torch.tensor(0.0, device=pred_box_offsets.device)}

        pred_center_offsets = src_box_offsets_matched[:, :3]
        pred_log_dim_offsets = src_box_offsets_matched[:, 3:6]
        pred_yaw_offsets = src_box_offsets_matched[:, 6:7]

        reconstructed_centers = ref_boxes_actual_dims_matched[:, :3] + pred_center_offsets
        reconstructed_log_dims = ref_boxes_log_dims_matched[:, 3:6] + pred_log_dim_offsets
        reconstructed_actual_dims = torch.exp(reconstructed_log_dims)
        reconstructed_yaws = ref_boxes_actual_dims_matched[:, 6:7] + pred_yaw_offsets
        reconstructed_yaws = (reconstructed_yaws + math.pi) % (2 * math.pi) - math.pi

        pred_boxes_reconstructed_for_giou = torch.cat(
            (reconstructed_centers, reconstructed_actual_dims, reconstructed_yaws), dim=-1
        )

        if pred_boxes_reconstructed_for_giou.shape[0] > 0 :
            giou_matrix_for_matched_pairs = generalized_bev_iou_pytorch(
                pred_boxes_reconstructed_for_giou, 
                ref_boxes_actual_dims_matched
            )
            giou_values_tensor = torch.diag(giou_matrix_for_matched_pairs)
            loss_giou = (1.0 - giou_values_tensor).sum() / num_total_boxes if num_total_boxes > 0 else torch.tensor(0.0, device=pred_box_offsets.device)
        else:
            loss_giou = torch.tensor(0.0, device=pred_box_offsets.device)
        
        return {'loss_giou': loss_giou}

    def forward(self, 
                decoder_outputs: Dict[str, torch.Tensor],
                gt_labels_b: torch.Tensor,
                gt_boxes_b_log_dims: torch.Tensor,
                gt_boxes_b_actual_dims: torch.Tensor,
                gt_valid_mask_b: torch.Tensor
               ) -> Dict[str, torch.Tensor]:
        
        pred_logits = decoder_outputs['pred_logits']
        pred_box_offsets = decoder_outputs['pred_box_offsets']

        indices = self.matcher(pred_logits, 
                               pred_box_offsets,
                               gt_labels_b,
                               gt_boxes_b_actual_dims,
                               gt_boxes_b_log_dims
                               )    

        num_total_matched_boxes = sum(len(t[0]) for t in indices)
        num_total_matched_boxes_tensor = torch.as_tensor([num_total_matched_boxes], dtype=torch.float, device=pred_logits.device)
        num_total_matched_boxes_final = torch.clamp(num_total_matched_boxes_tensor, min=1).item()

        losses = {}
        for loss_type in self.losses:
            if loss_type == 'labels':
                losses.update(self.loss_labels(pred_logits, gt_labels_b, indices, num_total_matched_boxes_final))
            elif loss_type == 'boxes_l1_offset': # << KORREKTUR HIER
                losses.update(self.loss_boxes_l1_offset(pred_box_offsets, indices, num_total_matched_boxes_final))
            elif loss_type == 'giou_bev':
                losses.update(self.loss_boxes_giou(pred_box_offsets, gt_boxes_b_actual_dims, gt_boxes_b_log_dims, gt_labels_b, indices, num_total_matched_boxes_final))
            else:
                raise ValueError(f"Unbekannter Verlusttyp in self.losses: {loss_type}")
        return losses

if __name__ == '__main__':
    print("Running SetCriterion and HungarianMatcher example (Offset Prediction)...")
    
    config_file_path_loss_main = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'pipeline_c_modules.yaml')

    if not os.path.exists(config_file_path_loss_main):
        print(f"WARNUNG: Config file not found at {config_file_path_loss_main}. Using default test parameters.")
        cfg_dict_for_test_loss_main = { 
            "model": {"num_classes": 12, "num_queries": 10, "box_dim": 7, 
                      "pe_max_coord_val": 150.0, "center_offset_scale": 5.0},
            "loss": { "cost_class_weight": 2.0, "cost_bbox_l1_offset_weight": 5.0, "cost_giou_bev_weight": 2.0,
                      "eos_coefficient": 0.1, "losses_to_compute": ["labels", "boxes_l1_offset", "giou_bev"], 
                      "loss_weight_dict": {"loss_ce": 1.0, "loss_bbox_l1_offset": 5.0, "loss_giou": 2.0} }}
    else:
        from oft.utils.config import load_config 
        cfg_dict_for_test_loss_main = load_config(config_file_path_loss_main)
        if not isinstance(cfg_dict_for_test_loss_main, dict):
            cfg_dict_for_test_loss_main = OmegaConf.to_container(cfg_dict_for_test_loss_main, resolve=True)
        if 'center_offset_scale' not in cfg_dict_for_test_loss_main.get('model', {}):
            cfg_dict_for_test_loss_main.setdefault('model', {})['center_offset_scale'] = 5.0
        if 'cost_bbox_l1_weight' in cfg_dict_for_test_loss_main.get('loss', {}) and 'cost_bbox_l1_offset_weight' not in cfg_dict_for_test_loss_main.get('loss', {}):
            cfg_dict_for_test_loss_main.setdefault('loss', {})['cost_bbox_l1_offset_weight'] = cfg_dict_for_test_loss_main.get('loss').pop('cost_bbox_l1_weight')
        if 'loss_bbox_l1' in cfg_dict_for_test_loss_main.get('loss', {}).get('loss_weight_dict', {}) and 'loss_bbox_l1_offset' not in cfg_dict_for_test_loss_main.get('loss', {}).get('loss_weight_dict', {}):
             cfg_dict_for_test_loss_main.setdefault('loss', {}).setdefault('loss_weight_dict', {})['loss_bbox_l1_offset'] = cfg_dict_for_test_loss_main.get('loss').get('loss_weight_dict').pop('loss_bbox_l1')
        if 'boxes_l1' in cfg_dict_for_test_loss_main.get('loss', {}).get('losses_to_compute', []) and 'boxes_l1_offset' not in cfg_dict_for_test_loss_main.get('loss', {}).get('losses_to_compute', []):
            losses_list = cfg_dict_for_test_loss_main.setdefault('loss', {}).setdefault('losses_to_compute', [])
            cfg_dict_for_test_loss_main['loss']['losses_to_compute'] = [lc if lc != 'boxes_l1' else 'boxes_l1_offset' for lc in losses_list]


        print(f"Konfiguration für Loss-Test geladen von: {config_file_path_loss_main}")


    model_cfg_loss = cfg_dict_for_test_loss_main.get('model', {})
    loss_cfg_loss = cfg_dict_for_test_loss_main.get('loss', {})
     
    num_classes_cfg = model_cfg_loss.get('num_classes', 12)
    num_queries_cfg = model_cfg_loss.get('num_queries', 100)
    
    cost_class_cfg = loss_cfg_loss.get('cost_class_weight', 2.0)
    cost_bbox_l1_offset_cfg = loss_cfg_loss.get('cost_bbox_l1_offset_weight', 5.0) 
    cost_giou_bev_cfg = loss_cfg_loss.get('cost_giou_bev_weight', 2.0)
    center_offset_scale_matcher_cfg = model_cfg_loss.get('center_offset_scale', 5.0)

    eos_coef_cfg = loss_cfg_loss.get('eos_coefficient', 0.1)
    losses_to_compute_cfg = loss_cfg_loss.get('losses_to_compute', ["labels", "boxes_l1_offset", "giou_bev"])
    
    loss_weight_dict_cfg = loss_cfg_loss.get('loss_weight_dict', {"loss_ce": 1.0, "loss_bbox_l1_offset": 5.0, "loss_giou": 2.0})
    coord_norm_factor_cfg = model_cfg_loss.get('pe_max_coord_val', 150.0)
    center_offset_scale_criterion_cfg = model_cfg_loss.get('center_offset_scale', 5.0)

    batch_s = 2
    device = torch.device("cpu") 
    box_dim_internal = 7

    dummy_pred_logits = torch.rand(batch_s, num_queries_cfg, num_classes_cfg + 1, device=device)
    dummy_pred_box_offsets = torch.randn(batch_s, num_queries_cfg, box_dim_internal, device=device) 
    dummy_pred_box_offsets[..., :3] = torch.tanh(dummy_pred_box_offsets[..., :3]) * center_offset_scale_criterion_cfg
    dummy_pred_box_offsets[..., 3:6] = torch.randn(batch_s, num_queries_cfg, 3, device=device) * 0.1
    dummy_pred_box_offsets[..., 6:7] = torch.tanh(torch.randn(batch_s, num_queries_cfg, 1, device=device)) * (math.pi / 4)

    max_gt_objs_for_test = 5
    dummy_gt_labels_b = torch.randint(0, num_classes_cfg, (batch_s, max_gt_objs_for_test), device=device)
    
    dummy_gt_boxes_b_log_dims = torch.randn(batch_s, max_gt_objs_for_test, box_dim_internal, device=device)
    dummy_gt_boxes_b_log_dims[..., :3] = torch.rand(batch_s, max_gt_objs_for_test, 3, device=device) * 50 -25 
    dummy_gt_boxes_b_log_dims[..., 3:6] = torch.log(torch.rand(batch_s, max_gt_objs_for_test, 3, device=device) * 4 + 1) 
    dummy_gt_boxes_b_log_dims[..., 6:7] = (torch.rand(batch_s, max_gt_objs_for_test, 1, device=device) - 0.5) * (2*math.pi)

    dummy_gt_boxes_b_actual_dims = dummy_gt_boxes_b_log_dims.clone()
    dummy_gt_boxes_b_actual_dims[..., 3:6] = torch.exp(dummy_gt_boxes_b_log_dims[..., 3:6])
    
    dummy_gt_valid_mask_b = torch.ones((batch_s, max_gt_objs_for_test), dtype=torch.bool, device=device)
    if max_gt_objs_for_test > 2: 
        dummy_gt_labels_b[0, -2:] = -1 
        dummy_gt_valid_mask_b[0, -2:] = False

    print(f"\nVerwendete Loss-Parameter (Offset-Version):")
    print(f"  Matcher Kosten: class={cost_class_cfg}, bbox_l1_offset={cost_bbox_l1_offset_cfg}, giou_bev={cost_giou_bev_cfg}")
    print(f"  Matcher Center Offset Scale (für L1-Kosten-Normalisierung, falls verwendet): {center_offset_scale_matcher_cfg}")
    print(f"  Criterion: eos_coef={eos_coef_cfg}, center_offset_scale (für Rekonstruktion)={center_offset_scale_criterion_cfg}")
    print(f"  Losses to compute: {losses_to_compute_cfg}")
    print(f"  Loss weights: {loss_weight_dict_cfg}")

    matcher_instance = HungarianMatcher(
        cost_class=cost_class_cfg, 
        cost_bbox_l1_offset=cost_bbox_l1_offset_cfg, 
        cost_giou_bev=cost_giou_bev_cfg,
        center_offset_scale_for_matcher_cost=center_offset_scale_matcher_cfg
    )
    criterion_instance = SetCriterion(
        num_classes=num_classes_cfg, matcher=matcher_instance, eos_coef=eos_coef_cfg,
        losses=losses_to_compute_cfg, 
        weight_dict=loss_weight_dict_cfg,
        coord_normalization_factor=coord_norm_factor_cfg, 
        center_offset_scale=center_offset_scale_criterion_cfg
    ).to(device)
    criterion_instance._print_count_l1_loss = 0 

    print("\nTeste SetCriterion.forward() mit Offsets:")
    decoder_outputs_test = {
        "pred_logits": dummy_pred_logits,
        "pred_box_offsets": dummy_pred_box_offsets 
    }
    calculated_losses = criterion_instance(
        decoder_outputs=decoder_outputs_test,
        gt_labels_b=dummy_gt_labels_b,
        gt_boxes_b_log_dims=dummy_gt_boxes_b_log_dims,     
        gt_boxes_b_actual_dims=dummy_gt_boxes_b_actual_dims, 
        gt_valid_mask_b=dummy_gt_valid_mask_b
    )

    print("\nBerechnete Verluste (ungewichtet):")
    for loss_name, loss_value in calculated_losses.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")
     
    print("\nSetCriterion and HungarianMatcher (Offset Prediction) example run successful.")