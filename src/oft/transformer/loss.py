import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional, Tuple
import numpy as np
import math

# NEU: Importiere Shapely für präzise IoU-Berechnung
from shapely.geometry import Polygon

# --- NEU: Shapely-basierte Hilfsfunktionen (aus deiner nms_3d.py übernommen) ---
def _box_to_polygon_shapely(box: np.ndarray) -> Polygon:
    """Konvertiert eine 7D-Box in ein Shapely Polygon für BEV IoU."""
    if not isinstance(box, np.ndarray) or box.shape[0] < 7:
        return Polygon()
    x, y, _, box_w, box_l, _, yaw = box[:7]
    # In Shapely wird oft l=x-dim, w=y-dim verwendet, passe dies ggf. an deine Box-Definition an
    dx, dy = box_l / 2.0, box_w / 2.0
    corners = np.array([
        [dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]
    ], dtype=np.float64)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rot_mat = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float64)
    corners_rot = corners.dot(rot_mat.T)
    corners_translated = corners_rot + np.array([x, y], dtype=np.float64)
    poly = Polygon(corners_translated)
    return poly

def _bev_iou_shapely_pair(box1: np.ndarray, box2: np.ndarray) -> float:
    """Berechnet die BEV IoU für ein einzelnes Boxenpaar mit Shapely."""
    poly1 = _box_to_polygon_shapely(box1)
    poly2 = _box_to_polygon_shapely(box2)
    if not poly1.is_valid or not poly2.is_valid or poly1.is_empty or poly2.is_empty:
        return 0.0
    
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.area + poly2.area - intersection_area
    if union_area < 1e-6:
        return 0.0
    
    return intersection_area / union_area

def _calculate_iou_matrix_shapely_np(boxes1_np: np.ndarray, boxes2_np: np.ndarray) -> np.ndarray:
    """Berechnet die IoU-Matrix für zwei Sätze von Boxen mit Shapely."""
    num_boxes1 = boxes1_np.shape[0]
    num_boxes2 = boxes2_np.shape[0]
    iou_matrix = np.zeros((num_boxes1, num_boxes2), dtype=np.float32)
    for i in range(num_boxes1):
        for j in range(num_boxes2):
            iou_matrix[i, j] = _bev_iou_shapely_pair(boxes1_np[i], boxes2_np[j])
    return iou_matrix

# --- Ende der Shapely-Funktionen ---

# --- Bisherige PyTorch-basierte Hilfsfunktionen (jetzt nur noch für Enclosing Box) ---
def get_bev_corners_pytorch(boxes_7d: torch.Tensor) -> torch.Tensor:
    if boxes_7d.ndim == 1:
        boxes_7d = boxes_7d.unsqueeze(0)
    centers_x, centers_y = boxes_7d[:, 0], boxes_7d[:, 1]
    widths, lengths = boxes_7d[:, 3], boxes_7d[:, 4]
    yaws = boxes_7d[:, 6]
    half_lengths, half_widths = lengths / 2.0, widths / 2.0
    corners_local = torch.tensor([[1, 1], [1, -1], [-1, -1], [-1, 1]], device=boxes_7d.device, dtype=boxes_7d.dtype)
    corners_local = corners_local * torch.stack([half_lengths, half_widths], dim=1).unsqueeze(1)
    
    cos_yaw, sin_yaw = torch.cos(yaws), torch.sin(yaws)
    rot_matrix = torch.stack([cos_yaw, -sin_yaw, sin_yaw, cos_yaw], dim=1).view(-1, 2, 2)
    
    corners_rotated = torch.bmm(corners_local, rot_matrix)
    corners_world = corners_rotated + torch.stack([centers_x, centers_y], dim=1).unsqueeze(1)
    return corners_world

def calculate_enclosing_box_area_bev_pytorch(corners1: torch.Tensor, corners2: torch.Tensor) -> torch.Tensor:
    """Behält die PyTorch-Implementierung für die umschließende Box bei."""
    # Erweitere die Dimensionen für das Broadcasting über alle Paare
    all_corners = torch.cat((corners1.unsqueeze(1).expand(-1, corners2.shape[0], -1, -1),
                             corners2.unsqueeze(0).expand(corners1.shape[0], -1, -1, -1)), dim=2)
    min_coords, _ = torch.min(all_corners, dim=2)
    max_coords, _ = torch.max(all_corners, dim=2)
    return (max_coords[..., 0] - min_coords[..., 0]) * (max_coords[..., 1] - min_coords[..., 1])


def generalized_bev_iou_shapely(boxes1_7d_actual_dims: torch.Tensor, boxes2_7d_actual_dims: torch.Tensor) -> torch.Tensor:
    """
    Berechnet die Generalisierte BEV IoU.
    - Verwendet Shapely für eine präzise, rotationsinvariante IoU.
    - Verwendet PyTorch für die umschließende Box, um die Differenzierbarkeit zu erhalten.
    """
    eps = 1e-7
    
    boxes1_np = boxes1_7d_actual_dims.detach().cpu().numpy()
    boxes2_np = boxes2_7d_actual_dims.detach().cpu().numpy()
    
    # Verwende torch.py_function, um die NumPy/Shapely-Funktion in den Graphen zu integrieren.
    # Beachte: Dies ist nicht differenzierbar, was für den GIoU-Term oft akzeptiert wird.
    iou_matrix_np = _calculate_iou_matrix_shapely_np(boxes1_np, boxes2_np)
    iou_matrix = torch.from_numpy(iou_matrix_np).to(boxes1_7d_actual_dims.device)

    areas1 = boxes1_7d_actual_dims[:, 3] * boxes1_7d_actual_dims[:, 4]
    areas2 = boxes2_7d_actual_dims[:, 3] * boxes2_7d_actual_dims[:, 4]
    corners1 = get_bev_corners_pytorch(boxes1_7d_actual_dims)
    corners2 = get_bev_corners_pytorch(boxes2_7d_actual_dims)
    
    intersection_area = iou_matrix * (areas1.unsqueeze(1) + areas2.unsqueeze(0) - iou_matrix * areas2.unsqueeze(0))
    union_area = areas1.unsqueeze(1) + areas2.unsqueeze(0) - intersection_area
    union_area = torch.clamp(union_area, min=eps)
    
    enclosing_area = calculate_enclosing_box_area_bev_pytorch(corners1, corners2)
    
    giou = iou_matrix - (enclosing_area - union_area) / (enclosing_area + eps)
    return torch.clamp(giou, min=-1.0, max=1.0)


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
            # BUGFIX: Holen der korrekten log_dims für das aktuelle Sample
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

            target_offsets_zeros_for_cdist = torch.zeros(num_gt_i, current_pred_box_offsets_i.shape[-1], device=pred_logits.device, dtype=current_pred_box_offsets_i.dtype)
            cost_bbox_l1_offset_matrix = torch.cdist(current_pred_box_offsets_i, target_offsets_zeros_for_cdist, p=1)
            
            # Rekonstruktion für GIoU
            recon_centers_all = current_gt_boxes_actual_dims_i[:, :3].unsqueeze(0) + current_pred_box_offsets_i[:, :3].unsqueeze(1)
            recon_log_dims_all = current_gt_boxes_log_dims_i[:, 3:6].unsqueeze(0) + current_pred_box_offsets_i[:, 3:6].unsqueeze(1)
            recon_actual_dims_all = torch.exp(recon_log_dims_all)
            recon_yaws_all = current_gt_boxes_actual_dims_i[:, 6].unsqueeze(0) + current_pred_box_offsets_i[:, 6].unsqueeze(1)
            recon_yaws_all = (recon_yaws_all + math.pi) % (2 * math.pi) - math.pi
            
            num_pred, num_gt = recon_centers_all.shape[:2]
            
            cost_giou_matrix = torch.zeros(num_pred, num_gt, device=pred_logits.device)
            # Schleife über GTs, da die Shapely-Funktion nicht den ganzen Batch verarbeiten kann
            for gt_idx in range(num_gt):
                pred_boxes_for_one_gt = torch.cat([
                    recon_centers_all[:, gt_idx, :],
                    recon_actual_dims_all[:, gt_idx, :],
                    recon_yaws_all[:, gt_idx].unsqueeze(-1)
                ], dim=-1)
                
                gt_box_single = current_gt_boxes_actual_dims_i[gt_idx].unsqueeze(0)
                giou_col = generalized_bev_iou_shapely(pred_boxes_for_one_gt, gt_box_single)
                cost_giou_matrix[:, gt_idx] = 1.0 - giou_col.squeeze(-1)

            C = (self.cost_class * cost_class_matrix +
                 self.cost_bbox_l1_offset * cost_bbox_l1_offset_matrix +
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
                 coord_normalization_factor: float = 1.0, 
                 center_offset_scale: float = 5.0
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
            target_classes_o = torch.cat(target_classes_o_list)
            if target_classes_o.numel() == batch_indices_for_preds.numel():
                target_classes[batch_indices_for_preds, matched_pred_indices_batch] = target_classes_o

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
            giou_matrix_for_matched_pairs = generalized_bev_iou_shapely(
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
            elif loss_type == 'boxes_l1_offset':
                losses.update(self.loss_boxes_l1_offset(pred_box_offsets, indices, num_total_matched_boxes_final))
            elif loss_type == 'giou_bev':
                losses.update(self.loss_boxes_giou(pred_box_offsets, gt_boxes_b_actual_dims, gt_boxes_b_log_dims, gt_labels_b, indices, num_total_matched_boxes_final))
            else:
                raise ValueError(f"Unbekannter Verlusttyp in self.losses: {loss_type}")
        return losses