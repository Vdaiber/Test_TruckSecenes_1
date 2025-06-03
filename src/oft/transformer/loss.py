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
    eps = 1e-7 

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
                pred_boxes: torch.Tensor, 
                gt_labels_b: torch.Tensor,
                gt_boxes_b: torch.Tensor  
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
            
            giou_values = generalized_bev_iou_pytorch(current_pred_boxes_i, current_gt_boxes_i)
            cost_giou_matrix = 1.0 - giou_values 

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
        
        if not target_boxes_list or src_boxes.numel() == 0: 
            return {'loss_bbox_l1': torch.tensor(0.0, device=pred_boxes.device, dtype=pred_boxes.dtype)} 

        target_boxes = torch.cat(target_boxes_list, dim=0) 

        if src_boxes.shape[0] != target_boxes.shape[0]:
            print(f"WARNUNG loss_boxes_l1: Shape-Mismatch nach dem Sammeln. src: {src_boxes.shape}, tgt: {target_boxes.shape}")
            return {'loss_bbox_l1': torch.tensor(0.0, device=pred_boxes.device, dtype=pred_boxes.dtype)} 

        # --- NORMALISIERUNG DER ZENTREN FÜR L1-LOSS ---
        src_centers_normalized = src_boxes[..., :3] / self.coord_normalization_factor
        tgt_centers_normalized = target_boxes[..., :3] / self.coord_normalization_factor
        
        loss_centers_l1 = F.l1_loss(src_centers_normalized, tgt_centers_normalized, reduction='none')
        loss_dims_yaw_l1 = F.l1_loss(src_boxes[..., 3:], target_boxes[..., 3:], reduction='none')
        
        loss_bbox_l1_all_params = torch.cat((loss_centers_l1, loss_dims_yaw_l1), dim=-1)
        # --- ENDE NORMALISIERUNG ---
        
        # --- DEBUG PRINT ---
        if self._print_count_l1_loss < 3: 
            print(f"\n--- DEBUG loss_boxes_l1 (Call {self._print_count_l1_loss + 1}) ---")
            print(f"  coord_normalization_factor: {self.coord_normalization_factor}")
            print(f"  Number of matched boxes in batch: {src_boxes.shape[0]}")
            if src_boxes.shape[0] > 0:
                num_to_print = min(src_boxes.shape[0], 3) 
                print(f"  Beispiele für gematchte Paare (bis zu {num_to_print}):")
                for k in range(num_to_print):
                    print(f"    Paar {k+1}:")
                    print(f"      Pred Box (src_boxes[{k}]): {np.round(src_boxes[k].detach().cpu().numpy(), 2)}")
                    print(f"      GT Box   (target_boxes[{k}]): {np.round(target_boxes[k].detach().cpu().numpy(), 2)}")
                    print(f"      Pred Centers NORM: {np.round(src_centers_normalized[k].detach().cpu().numpy(), 4)}")
                    print(f"      GT Centers   NORM: {np.round(tgt_centers_normalized[k].detach().cpu().numpy(), 4)}")
                    print(f"      L1 Diff Centers NORM: {np.round(torch.abs(src_centers_normalized[k] - tgt_centers_normalized[k]).detach().cpu().numpy(), 4)}")
                    print(f"      L1 Diff Dims/Yaw  : {np.round(torch.abs(src_boxes[k, 3:] - target_boxes[k, 3:]).detach().cpu().numpy(), 2)}")
            self._print_count_l1_loss += 1
        # --- ENDE DEBUG PRINT ---

        losses = {}
        losses['loss_bbox_l1'] = loss_bbox_l1_all_params.sum() / num_total_boxes 
        return losses

    def loss_boxes_giou(self, 
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
        
        if not target_boxes_list or src_boxes.numel() == 0: 
            return {'loss_giou': torch.tensor(0.0, device=pred_boxes.device, dtype=pred_boxes.dtype)} 
            
        target_boxes = torch.cat(target_boxes_list, dim=0)
        
        if src_boxes.shape[0] != target_boxes.shape[0]:
            print(f"WARNUNG loss_boxes_giou: Shape-Mismatch nach dem Sammeln. src: {src_boxes.shape}, tgt: {target_boxes.shape}")
            return {'loss_giou': torch.tensor(0.0, device=pred_boxes.device, dtype=pred_boxes.dtype)} 

        giou_values_per_pair = []
        if src_boxes.numel() > 0 and target_boxes.numel() > 0 : 
            if src_boxes.shape[0] > 0 and target_boxes.shape[0] > 0 :
                for k_pair in range(src_boxes.shape[0]):
                    giou_val = generalized_bev_iou_pytorch(src_boxes[k_pair].unsqueeze(0), 
                                                        target_boxes[k_pair].unsqueeze(0))
                    giou_values_per_pair.append(giou_val.squeeze())
        
        losses = {}
        if giou_values_per_pair:
            giou_values_tensor = torch.stack(giou_values_per_pair)
            loss_giou = (1.0 - giou_values_tensor).sum() / num_total_boxes 
            losses['loss_giou'] = loss_giou
        else:
            losses['loss_giou'] = torch.tensor(0.0, device=pred_boxes.device, dtype=pred_boxes.dtype)
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
            elif loss_type == 'giou_bev':
                losses.update(self.loss_boxes_giou(pred_boxes, gt_boxes_b, gt_labels_b, indices, num_total_matched_boxes))
            else:
                raise ValueError(f"Unbekannter Verlusttyp in self.losses: {loss_type}")
        return losses

if __name__ == '__main__':
    print("Running SetCriterion and HungarianMatcher example with config-loaded parameters...")
    from oft.utils.config import load_config 

    config_file_path = "config/pipeline_c_modules.yaml"
    cfg_dict_for_test = None
    if not os.path.exists(config_file_path):
        alt_config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'pipeline_c_modules.yaml')
        if os.path.exists(alt_config_path):
            config_file_path = alt_config_path
        else:
            print(f"ERROR: Config file not found at {config_file_path} or {alt_config_path}")
            config_file_path = None 
            
    if config_file_path:
        try:
            cfg_dict_for_test = load_config(config_file_path)
            print(f"Konfiguration für Loss-Test geladen von: {os.path.abspath(config_file_path)}")
        except Exception as e:
            print(f"Fehler beim Laden der Konfiguration für Loss-Test: {e}")
            cfg_dict_for_test = None 
            
    if cfg_dict_for_test is None:
        print("Using fallback configuration for loss test.")
        cfg_dict_for_test = {
            "model": {"num_classes": 5, "num_queries": 10, "box_dim": 7, "pe_max_coord_val": 150.0}, 
            "loss": {
                "cost_class_weight": 1.0, "cost_bbox_l1_weight": 5.0, "cost_giou_bev_weight": 2.0,
                "eos_coefficient": 0.1, "losses_to_compute": ["labels", "boxes_l1", "giou_bev"],
                "loss_weight_dict": {"loss_ce": 1.0, "loss_bbox_l1": 5.0, "loss_giou": 2.0},
            }
        }
    full_pipeline_config = OmegaConf.create(cfg_dict_for_test)
    
    model_cfg = full_pipeline_config.get('model', OmegaConf.create({})) 
    loss_cfg = full_pipeline_config.get('loss', OmegaConf.create({}))
    
    num_classes_cfg = model_cfg.get('num_classes', 5) 
    num_queries_cfg = model_cfg.get('num_queries', 10)
    box_dim_cfg = model_cfg.get('box_dim', 7)
    cost_class_cfg = loss_cfg.get('cost_class_weight', 1.0)
    cost_bbox_l1_cfg = loss_cfg.get('cost_bbox_l1_weight', 5.0) 
    cost_giou_bev_cfg = loss_cfg.get('cost_giou_bev_weight', 2.0) 
    eos_coef_cfg = loss_cfg.get('eos_coefficient', 0.1)
    losses_to_compute_cfg = list(OmegaConf.to_container(loss_cfg.get('losses_to_compute', ['labels', 'boxes_l1', 'giou_bev']), resolve=True))
    loss_weight_dict_cfg = dict(OmegaConf.to_container(loss_cfg.get('loss_weight_dict', {'loss_ce': 1.0, 'loss_bbox_l1': 5.0, 'loss_giou': 2.0}), resolve=True))
    coord_norm_factor_cfg = model_cfg.get('pe_max_coord_val', 150.0) # Holen aus model_cfg für Test


    batch_size = 2
    device = torch.device("cpu") 
    dummy_pred_logits = torch.rand(batch_size, num_queries_cfg, num_classes_cfg + 1, device=device) 
    
    raw_dummy_pred_boxes = torch.rand(batch_size, num_queries_cfg, box_dim_cfg, device=device)
    dummy_pred_cxcycz = (raw_dummy_pred_boxes[..., :3] * 100) - 50 
    dummy_pred_wlh = torch.exp(raw_dummy_pred_boxes[..., 3:6] * 0.5) 
    dummy_pred_yaw = torch.tanh(raw_dummy_pred_boxes[..., 6:7]) * math.pi 
    dummy_pred_boxes = torch.cat((dummy_pred_cxcycz, dummy_pred_wlh, dummy_pred_yaw), dim=-1)

    max_gt_objs_in_batch_for_test = 3 
    dummy_gt_labels_b = torch.full((batch_size, max_gt_objs_in_batch_for_test), -1, dtype=torch.long, device=device) 
    dummy_gt_boxes_b = torch.zeros((batch_size, max_gt_objs_in_batch_for_test, box_dim_cfg), dtype=torch.float32, device=device)

    if max_gt_objs_in_batch_for_test >=2:
        dummy_gt_labels_b[0, :2] = torch.tensor([0, 1], device=device) 
        dummy_gt_boxes_b[0, 0, :3] = dummy_pred_boxes[0,0,:3].clone().detach() + torch.randn_like(dummy_pred_boxes[0,0,:3]) * 5 
        dummy_gt_boxes_b[0, 0, 3:6] = dummy_pred_boxes[0,0,3:6].clone().detach() * (torch.rand_like(dummy_pred_boxes[0,0,3:6])*0.4 + 0.8) 
        dummy_gt_boxes_b[0, 0, 6] = dummy_pred_boxes[0,0,6].clone().detach() + (torch.rand_like(dummy_pred_boxes[0,0,6:7])*0.4 - 0.2).squeeze() 

        dummy_gt_boxes_b[0, 1, :3] = dummy_pred_boxes[0,1,:3].clone().detach() - torch.randn_like(dummy_pred_boxes[0,1,:3]) * 5
        dummy_gt_boxes_b[0, 1, 3:6] = dummy_pred_boxes[0,1,3:6].clone().detach() * (torch.rand_like(dummy_pred_boxes[0,1,3:6])*0.3 + 0.9)
        dummy_gt_boxes_b[0, 1, 6] = dummy_pred_boxes[0,1,6].clone().detach() + (torch.rand_like(dummy_pred_boxes[0,1,6:7])*0.6 - 0.3).squeeze()
        dummy_gt_boxes_b[0, :2, 3:6] = torch.clamp(dummy_gt_boxes_b[0, :2, 3:6], min=0.5) 

    if max_gt_objs_in_batch_for_test >=1:
        dummy_gt_labels_b[1, :1] = torch.tensor([2], device=device) 
        dummy_gt_boxes_b[1, 0, :] = dummy_pred_boxes[1, 0, :].clone().detach() + torch.rand_like(dummy_pred_boxes[1,0,:]) * 0.2
        dummy_gt_boxes_b[1, :1, 3:6] = torch.clamp(dummy_gt_boxes_b[1, :1, 3:6], min=0.5)


    print(f"\nVerwendete Loss-Parameter (aus Config oder Fallback für Test):")
    print(f"  num_classes (ohne BG): {num_classes_cfg}, num_queries: {num_queries_cfg}, box_dim: {box_dim_cfg}")
    print(f"  Matcher Kosten: class={cost_class_cfg}, bbox_l1={cost_bbox_l1_cfg}, giou_bev={cost_giou_bev_cfg}")
    print(f"  Criterion: eos_coef={eos_coef_cfg}, losses={losses_to_compute_cfg}, weights={loss_weight_dict_cfg}")
    print(f"  Criterion coord_normalization_factor: {coord_norm_factor_cfg}")


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
        weight_dict=loss_weight_dict_cfg,
        coord_normalization_factor=coord_norm_factor_cfg 
    ).to(device)
    criterion._print_count_l1_loss = 0 

    print("\nTeste HungarianMatcher separat:")
    matched_indices = matcher_instance(dummy_pred_logits, dummy_pred_boxes, dummy_gt_labels_b, dummy_gt_boxes_b)
    for i_sample, (pred_idx, gt_idx) in enumerate(matched_indices):
        print(f"  Sample {i_sample}: Matched Pred Indices: {pred_idx.tolist()}, Matched GT Indices (relativ zu validen GTs): {gt_idx.tolist()}")

    print("\nTeste SetCriterion.forward() mit Debug-Ausgaben in L1-Loss:")
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
        for k_loss, v_loss in calculated_losses.items(): 
            if k_loss in criterion.weight_dict: 
                total_weighted_loss += v_loss * criterion.weight_dict[k_loss]
            else:
                print(f"    WARNUNG: Kein Gewicht für '{k_loss}' im weight_dict des Kriteriums gefunden.")
    print(f"  Gewichteter Gesamtverlust: {total_weighted_loss.item():.4f}")
    
    print("\nSetCriterion and HungarianMatcher example run successful.")