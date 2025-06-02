# src/oft/transformer/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional, Tuple
import numpy as np

import os
from oft.utils.config import load_config
from omegaconf import OmegaConf # Für die Konvertierung von DictConfig und Erstellung

def placeholder_bev_iou_loss(boxes1_7d: torch.Tensor, boxes2_7d: torch.Tensor) -> torch.Tensor:
    """
    Platzhalter für BEV IoU / GIoU Loss.
    Aktuell implementiert als eine skalierte L1-Distanz der 2D-Zentren,
    um einen Wert ungleich Null zu liefern, der aber nicht die echte IoU ist.
    Sollte durch eine korrekte IoU/GIoU-Berechnung im BEV ersetzt werden.
    Clamped den Wert auf maximal 2.0, um den DETR GIoU-Loss-Bereich (-1 bis 1 für IoU, 0 bis 2 für GIoU-Loss)
    grob zu simulieren, wobei höhere Werte hier schlechter sind.
    """
    if boxes1_7d.numel() == 0 or boxes2_7d.numel() == 0:
        return torch.tensor(0.0, device=boxes1_7d.device)
    
    # Simuliere einen "Distanz"-basierten Loss, der größer ist, je weiter die Boxen entfernt sind.
    # Dies ist KEIN IoU/GIoU, dient nur dazu, dass der Code läuft.
    # Verwende nur cx, cy (Indizes 0, 1)
    cost = torch.cdist(boxes1_7d[:, :2], boxes2_7d[:, :2], p=1) / 10.0 # Skalierungsfaktor, um Werte < 2 zu bekommen
    return torch.clamp(cost, max=2.0) # GIoU Loss ist typischerweise [0, 2]


class HungarianMatcher(nn.Module):
    """
    Matcher, der Vorhersagen den Ground-Truth-Boxen zuordnet.
    Basierend auf dem DETR-Matcher.
    """
    def __init__(self,
                 cost_class: float = 1.0,      # Gewicht für Klassifikationskosten
                 cost_bbox_l1: float = 1.0,    # Gewicht für L1-Distanz der Boxen
                 cost_giou_bev: float = 1.0):  # Gewicht für GIoU-Kosten (aktuell Placeholder)
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox_l1 = cost_bbox_l1
        self.cost_giou_bev = cost_giou_bev
        if cost_class == 0 and cost_bbox_l1 == 0 and cost_giou_bev == 0:
            raise ValueError("Alle Kosten-Gewichte im Matcher dürfen nicht null sein.")

    @torch.no_grad() # Wichtig: Matching erfordert keine Gradienten
    def forward(self,
                pred_logits: torch.Tensor,  # (Batch, NumQueries, NumClasses + 1)
                pred_boxes: torch.Tensor,   # (Batch, NumQueries, BoxDim)
                gt_labels_b: torch.Tensor,    # (Batch, MaxNumGTObjects) - gepadded mit -1
                gt_boxes_b: torch.Tensor      # (Batch, MaxNumGTObjects, BoxDim) - gepadded
               ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Führt das Matching für einen Batch durch.
        Args:
            pred_logits: Klassifikations-Logits der Vorhersagen.
            pred_boxes: Vorhergesagte Boxen (7D: cx,cy,cz,w,l,h,yaw).
            gt_labels_b: Ground-Truth Labels pro Sample, gepadded mit -1.
            gt_boxes_b: Ground-Truth Boxen pro Sample, gepadded.
        Returns:
            Eine Liste von Tupeln (pred_indices, gt_indices) für jedes Sample im Batch.
            Diese Indizes beziehen sich auf die *ungepaddeten* GTs.
        """
        batch_size, num_queries = pred_logits.shape[:2]
        indices = [] # Liste, um die (pred_idx, gt_idx) Paare pro Batch-Element zu speichern

        # Iteriere über jedes Sample im Batch
        for i in range(batch_size):
            # Filtere gepaddete GT-Elemente heraus für dieses Sample
            valid_gt_mask_i = (gt_labels_b[i] >= 0) # True für valide GTs
            
            # Wenn keine validen GTs für dieses Sample vorhanden sind, gibt es keine Matches
            if not valid_gt_mask_i.any(): 
                indices.append((torch.tensor([], dtype=torch.long, device=pred_logits.device),
                                torch.tensor([], dtype=torch.long, device=pred_logits.device)))
                continue

            current_gt_labels_i = gt_labels_b[i][valid_gt_mask_i] # Nur valide GT-Labels
            current_gt_boxes_i = gt_boxes_b[i][valid_gt_mask_i]   # Nur valide GT-Boxen
            num_gt_i = current_gt_labels_i.shape[0]

            # Wenn nach dem Filtern keine GTs übrig bleiben (sollte durch .any() oben abgedeckt sein)
            if num_gt_i == 0: 
                indices.append((torch.tensor([], dtype=torch.long, device=pred_logits.device),
                                torch.tensor([], dtype=torch.long, device=pred_logits.device)))
                continue

            # Vorhersagen für das aktuelle Sample
            current_pred_logits_i = pred_logits[i] # (NumQueries, NumClasses + 1)
            current_pred_boxes_i = pred_boxes[i]   # (NumQueries, BoxDim)

            # --- Kostenberechnung ---
            # Klassifikationskosten: -P(gt_class | pred)
            # Nutze die Logits direkt oder die Wahrscheinlichkeiten nach Softmax. DETR verwendet oft Logits.
            # Hier verwenden wir Wahrscheinlichkeiten für die Kostenmatrix.
            prob = current_pred_logits_i.softmax(-1) # (NumQueries, NumClasses + 1)
            # Kosten sind -Wahrscheinlichkeit der korrekten Klasse. Indiziere mit den GT-Labels.
            cost_class_matrix = -prob[:, current_gt_labels_i] # (NumQueries, NumGT_i)

            # L1-Kosten für Bounding Boxes
            cost_bbox_l1_matrix = torch.cdist(current_pred_boxes_i, current_gt_boxes_i, p=1) # (NumQueries, NumGT_i)
            
            # GIoU-Kosten (aktuell Placeholder)
            # Wichtig: placeholder_bev_iou_loss erwartet (N, 7) und (M, 7) und gibt (N, M) zurück
            cost_giou_matrix = placeholder_bev_iou_loss(current_pred_boxes_i, current_gt_boxes_i) # (NumQueries, NumGT_i)

            # Gesamtkostenmatrix
            C = (self.cost_class * cost_class_matrix +
                 self.cost_bbox_l1 * cost_bbox_l1_matrix +
                 self.cost_giou_bev * cost_giou_matrix)
            
            # Konvertiere zu NumPy für linear_sum_assignment
            C_np = C.detach().cpu().numpy()
            
            # Führe das Matching durch
            row_ind, col_ind = linear_sum_assignment(C_np) # row_ind sind Prädiktions-Indizes, col_ind sind GT-Indizes
            
            # Speichere die gematchten Indizes als Tensoren
            indices.append((torch.as_tensor(row_ind, dtype=torch.long, device=pred_logits.device),
                            torch.as_tensor(col_ind, dtype=torch.long, device=pred_logits.device)))
        return indices


class SetCriterion(nn.Module):
    """
    Verlustfunktion für Transformer-basierte Objektdetektion.
    Berechnet einen Satz von Verlusten (Klassifikation, BBox L1, GIoU)
    basierend auf den Zuordnungen des HungarianMatchers.
    """
    def __init__(self, 
                 num_classes: int, 
                 matcher: HungarianMatcher,
                 eos_coef: float, # Gewicht für die "no object" Klasse
                 losses: List[str], # Liste der zu berechnenden Verluste, z.B. ['labels', 'boxes_l1', 'giou_bev']
                 weight_dict: Dict[str, float] # Gewichtung für jeden berechneten Verlust
                ): 
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef 
        self.losses = losses # Speichere die Liste der zu berechnenden Verluste
        self.weight_dict = weight_dict # Speichere das Dictionary mit den Gewichten

        # Erstelle Gewichte für CrossEntropyLoss (eos_coef für "no object" Klasse)
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef # Letzte Klasse ist "no object"
        self.register_buffer('empty_weight', empty_weight)

    def _get_src_permutation_idx(self, indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Erstellt Batch- und Quellen-Indizes für die gematchten Vorhersagen. """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Erstellt Batch- und Ziel-Indizes für die gematchten Ground-Truth-Objekte. """
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_labels(self, 
                    pred_logits: torch.Tensor, # (Batch, NumQueries, NumClasses + 1)
                    gt_labels_b: torch.Tensor,   # (Batch, MaxNumGTObjects), gepadded mit -1
                    indices: List[Tuple[torch.Tensor, torch.Tensor]], # Output des Matchers
                    num_total_boxes: int # Gesamtzahl der gematchten Boxen über den Batch
                   ) -> Dict[str, torch.Tensor]:
        """Klassifikationsverlust (Cross Entropy)."""
        # Erstelle Ziel-Klassenlabels für alle Vorhersagen (Queries)
        # Initialisiere alle mit "no object" Klasse (self.num_classes)
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes,
                                    dtype=torch.long, device=pred_logits.device)
        
        # Hole die Batch- und Prädiktions-Indizes der gematchten Paare
        batch_indices_for_preds, matched_pred_indices_batch = self._get_src_permutation_idx(indices)
        
        # Erstelle eine Liste der GT-Labels für die gematchten Vorhersagen
        target_classes_o_list = [] 
        for i, (_, gt_idx_sample) in enumerate(indices): # (_, gt_idx_sample) sind die Indizes der gematchten GTs für Sample i
            valid_gt_labels_sample_i = gt_labels_b[i][gt_labels_b[i] >= 0] # Nur valide GT-Labels für dieses Sample
            if gt_idx_sample.numel() > 0 and valid_gt_labels_sample_i.numel() > 0:
                 # Stelle sicher, dass gt_idx_sample innerhalb der Grenzen von valid_gt_labels_sample_i liegt
                 if gt_idx_sample.max() < len(valid_gt_labels_sample_i):
                    target_classes_o_list.append(valid_gt_labels_sample_i[gt_idx_sample])
                # else:
                    # Dieser Fall sollte durch das Matching und die GT-Vorbereitung nicht oft auftreten,
                    # aber eine Warnung könnte hier nützlich sein.
                    # print(f"WARNUNG loss_labels: gt_idx_sample.max() ({gt_idx_sample.max()}) >= len(valid_gt_labels) ({len(valid_gt_labels_sample_i)})")

        
        if target_classes_o_list: # Nur wenn es Matches gab
            target_classes_o = torch.cat(target_classes_o_list)
            # Weise den gematchten Vorhersagen die korrekten GT-Klassen zu
            if target_classes_o.numel() == batch_indices_for_preds.numel(): # Sicherheitscheck
                target_classes[batch_indices_for_preds, matched_pred_indices_batch] = target_classes_o
            # else:
                # print(f"WARNUNG loss_labels: Anzahl gematchter GT-Labels ({target_classes_o.numel()}) != Anzahl gematchter Prädiktionen ({batch_indices_for_preds.numel()})")


        # Berechne Cross-Entropy-Verlust
        # pred_logits: (Batch, NumQueries, NumClasses + 1) -> Transponieren für CrossEntropy
        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes_l1(self, 
                      pred_boxes: torch.Tensor, # (Batch, NumQueries, BoxDim)
                      gt_boxes_b: torch.Tensor,   # (Batch, MaxNumGTObjects, BoxDim), gepadded
                      gt_labels_b: torch.Tensor,  # (Batch, MaxNumGTObjects), gepadded mit -1 (für Filterung)
                      indices: List[Tuple[torch.Tensor, torch.Tensor]],
                      num_total_boxes: int # Gesamtzahl gematchter Boxen
                     ) -> Dict[str, torch.Tensor]:
        """L1-Verlust für Bounding Boxes."""
        batch_indices_for_preds, matched_pred_indices_batch = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[batch_indices_for_preds, matched_pred_indices_batch] # Nur die gematchten Vorhersagen

        # Sammle die entsprechenden GT-Boxen
        target_boxes_list = []
        for i, (_, gt_idx_sample) in enumerate(indices): # gt_idx_sample bezieht sich auf die validen GTs
            valid_gt_mask_sample_i = (gt_labels_b[i] >= 0)
            valid_gt_boxes_sample_i = gt_boxes_b[i][valid_gt_mask_sample_i] # Nur valide GT-Boxen für dieses Sample
            if gt_idx_sample.numel() > 0 and valid_gt_boxes_sample_i.numel() > 0:
                if gt_idx_sample.max() < valid_gt_boxes_sample_i.shape[0]: # Sicherheitscheck
                    target_boxes_list.append(valid_gt_boxes_sample_i[gt_idx_sample])
        
        if not target_boxes_list : # Keine Matches im gesamten Batch
            return {'loss_bbox_l1': torch.tensor(0.0, device=pred_boxes.device)}

        target_boxes = torch.cat(target_boxes_list, dim=0) 

        # Sicherheitscheck, ob die Anzahl der gematchten Prädiktionen und GTs übereinstimmt
        if src_boxes.shape[0] != target_boxes.shape[0]: 
            # print(f"WARNUNG loss_boxes_l1: src_boxes.shape[0] ({src_boxes.shape[0]}) != target_boxes.shape[0] ({target_boxes.shape[0]})")
            return {'loss_bbox_l1': torch.tensor(0.0, device=pred_boxes.device)} # Oder Fehler werfen

        loss_bbox_l1 = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox_l1'] = loss_bbox_l1.sum() / num_total_boxes # Normiere durch Anzahl gematchter Boxen
        return losses

    def loss_boxes_giou(self, 
                        pred_boxes: torch.Tensor, # (Batch, NumQueries, BoxDim)
                        gt_boxes_b: torch.Tensor,   # (Batch, MaxNumGTObjects, BoxDim), gepadded
                        gt_labels_b: torch.Tensor,  # (Batch, MaxNumGTObjects), gepadded mit -1 (für Filterung)
                        indices: List[Tuple[torch.Tensor, torch.Tensor]],
                        num_total_boxes: int # Gesamtzahl gematchter Boxen
                       ) -> Dict[str, torch.Tensor]:
        """GIoU-Verlust für Bounding Boxes (aktuell mit Placeholder)."""
        batch_indices_for_preds, matched_pred_indices_batch = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[batch_indices_for_preds, matched_pred_indices_batch]

        target_boxes_list = []
        for i, (_, gt_idx_sample) in enumerate(indices):
            valid_gt_mask_sample_i = (gt_labels_b[i] >= 0)
            valid_gt_boxes_sample_i = gt_boxes_b[i][valid_gt_mask_sample_i]
            if gt_idx_sample.numel() > 0 and valid_gt_boxes_sample_i.numel() > 0:
                if gt_idx_sample.max() < valid_gt_boxes_sample_i.shape[0]:
                     target_boxes_list.append(valid_gt_boxes_sample_i[gt_idx_sample])
        
        if not target_boxes_list:
            return {'loss_giou': torch.tensor(0.0, device=pred_boxes.device)}
            
        target_boxes = torch.cat(target_boxes_list, dim=0)
        
        if src_boxes.shape[0] != target_boxes.shape[0]:
            # print(f"WARNUNG loss_boxes_giou: src_boxes.shape[0] ({src_boxes.shape[0]}) != target_boxes.shape[0] ({target_boxes.shape[0]})")
            return {'loss_giou': torch.tensor(0.0, device=pred_boxes.device)}

        # Verwende den Placeholder für GIoU-Loss paarweise
        giou_costs_per_pair = []
        if src_boxes.numel() > 0 and target_boxes.numel() > 0: # Stelle sicher, dass Tensoren nicht leer sind
            for k_pair in range(src_boxes.shape[0]): # Iteriere über jedes gematchte Paar
                # placeholder_bev_iou_loss erwartet (N, 7) und (M, 7)
                cost_val = placeholder_bev_iou_loss(src_boxes[k_pair].unsqueeze(0), 
                                                    target_boxes[k_pair].unsqueeze(0))
                giou_costs_per_pair.append(cost_val.squeeze()) # Squeeze, um Skalar zu erhalten
        
        losses = {}
        if giou_costs_per_pair:
            loss_giou_sum = torch.stack(giou_costs_per_pair).sum()
            losses['loss_giou'] = loss_giou_sum / num_total_boxes # Normiere
        else:
            losses['loss_giou'] = torch.tensor(0.0, device=pred_boxes.device)
        return losses

    def forward(self, 
                decoder_outputs: Dict[str, torch.Tensor], 
                gt_labels_b: torch.Tensor,   # (Batch, MaxNumGTObjects), gepadded mit -1
                gt_boxes_b: torch.Tensor      # (Batch, MaxNumGTObjects, BoxDim), gepadded
               ) -> Dict[str, torch.Tensor]:
        """
        Berechnet die Verluste.
        Args:
            decoder_outputs: Dictionary mit 'pred_logits' und 'pred_boxes'.
            gt_labels_b: Ground-Truth Labels pro Sample, gepadded mit -1.
            gt_boxes_b: Ground-Truth Boxen pro Sample, gepadded.
        Returns:
            Ein Dictionary mit den berechneten Verlusten (ungewichtet).
        """
        pred_logits = decoder_outputs['pred_logits']
        pred_boxes = decoder_outputs['pred_boxes']

        # Führe das Matching durch, um Vorhersagen den GTs zuzuordnen
        indices = self.matcher(pred_logits, pred_boxes, gt_labels_b, gt_boxes_b)

        # Berechne die Gesamtzahl der gematchten Boxen über den Batch (für die Normierung der Verluste)
        num_total_matched_boxes = sum(len(t[0]) for t in indices)
        num_total_matched_boxes = torch.as_tensor([num_total_matched_boxes], dtype=torch.float, device=pred_logits.device)
        # Verhindere Division durch Null, wenn keine Boxen gematcht wurden
        num_total_matched_boxes = torch.clamp(num_total_matched_boxes, min=1).item()


        losses = {}
        for loss_type in self.losses: # self.losses kommt aus der Konfiguration (z.B. ['labels', 'boxes_l1', 'giou_bev'])
            if loss_type == 'labels':
                losses.update(self.loss_labels(pred_logits, gt_labels_b, indices, num_total_matched_boxes))
            elif loss_type == 'boxes_l1':
                losses.update(self.loss_boxes_l1(pred_boxes, gt_boxes_b, gt_labels_b, indices, num_total_matched_boxes))
            elif loss_type == 'giou_bev': # KORRIGIERT: Behandle 'giou_bev'
                losses.update(self.loss_boxes_giou(pred_boxes, gt_boxes_b, gt_labels_b, indices, num_total_matched_boxes))
            # elif loss_type == 'boxes_giou': # Falls der Key in der Config 'boxes_giou' wäre
            #     losses.update(self.loss_boxes_giou(pred_boxes, gt_boxes_b, gt_labels_b, indices, num_total_matched_boxes))
            else:
                # Dieser Fall sollte jetzt nicht mehr eintreten, wenn 'giou_bev' korrekt behandelt wird.
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
            exit()
            
    try:
        cfg_dict = load_config(config_file_path)
        full_pipeline_config = OmegaConf.create(cfg_dict) 
        print(f"Konfiguration für Loss-Test geladen von: {os.path.abspath(config_file_path)}")
        
        model_cfg = full_pipeline_config.get('model', OmegaConf.create({})) 
        loss_cfg = full_pipeline_config.get('loss', OmegaConf.create({}))
        
        num_classes_cfg = model_cfg.get('num_classes', 28) 
        num_queries_cfg = model_cfg.get('num_queries', 100)
        box_dim_cfg = model_cfg.get('box_dim', 7)
        cost_class_cfg = loss_cfg.get('cost_class_weight', 1.0)
        cost_bbox_l1_cfg = loss_cfg.get('cost_bbox_l1_weight', 5.0)
        cost_giou_bev_cfg = loss_cfg.get('cost_giou_bev_weight', 2.0)
        eos_coef_cfg = loss_cfg.get('eos_coefficient', 0.1)
        losses_to_compute_cfg = list(OmegaConf.to_container(loss_cfg.get('losses_to_compute', ['labels', 'boxes_l1', 'giou_bev']), resolve=True))
        loss_weight_dict_cfg = dict(OmegaConf.to_container(loss_cfg.get('loss_weight_dict', {'loss_ce': 1.0, 'loss_bbox_l1': 5.0, 'loss_giou': 2.0}), resolve=True)) # Füge loss_giou hinzu

    except Exception as e:
        print(f"Fehler beim Laden der Konfiguration für Loss-Test: {e}")
        print("Verwende Standard-Fallback-Parameter.")
        num_classes_cfg = 5 
        num_queries_cfg = 10
        box_dim_cfg = 7
        cost_class_cfg = 1.0
        cost_bbox_l1_cfg = 1.0
        cost_giou_bev_cfg = 1.0
        eos_coef_cfg = 0.1
        losses_to_compute_cfg = ['labels', 'boxes_l1', 'giou_bev'] # Füge giou_bev hinzu
        loss_weight_dict_cfg = {'loss_ce': 1.0, 'loss_bbox_l1': 1.0, 'loss_giou': 1.0} # Füge loss_giou hinzu

    batch_size = 2
    dummy_pred_logits = torch.rand(batch_size, num_queries_cfg, num_classes_cfg + 1) 
    dummy_pred_boxes = torch.rand(batch_size, num_queries_cfg, box_dim_cfg) * 50 
    max_gt_objs_in_batch_for_test = 4 
    dummy_gt_labels_b = torch.full((batch_size, max_gt_objs_in_batch_for_test), -1, dtype=torch.long) 
    dummy_gt_boxes_b = torch.zeros((batch_size, max_gt_objs_in_batch_for_test, box_dim_cfg), dtype=torch.float32)
    dummy_gt_valid_mask_b = torch.zeros((batch_size, max_gt_objs_in_batch_for_test), dtype=torch.bool)

    if max_gt_objs_in_batch_for_test >= 3:
        dummy_gt_labels_b[0, :3] = torch.randint(0, num_classes_cfg, (3,))
        dummy_gt_boxes_b[0, :3, :] = torch.rand(3, box_dim_cfg) * 50
        dummy_gt_valid_mask_b[0, :3] = True
    if max_gt_objs_in_batch_for_test >= 2:
        dummy_gt_labels_b[1, :2] = torch.randint(0, num_classes_cfg, (2,))
        dummy_gt_boxes_b[1, :2, :] = torch.rand(2, box_dim_cfg) * 50
        dummy_gt_valid_mask_b[1, :2] = True
    
    print(f"\nVerwendete Loss-Parameter (aus Config oder Fallback):")
    print(f"  num_classes (ohne BG): {num_classes_cfg}, num_queries: {num_queries_cfg}, box_dim: {box_dim_cfg}")
    print(f"  Matcher Kosten: class={cost_class_cfg}, bbox_l1={cost_bbox_l1_cfg}, giou_bev={cost_giou_bev_cfg}")
    print(f"  Criterion: eos_coef={eos_coef_cfg}, losses={losses_to_compute_cfg}, weights={loss_weight_dict_cfg}")
    print(f"\nDummy Inputs für Loss-Berechnung:")
    print(f"  Pred Logits shape: {dummy_pred_logits.shape}")
    print(f"  Pred Boxes shape: {dummy_pred_boxes.shape}")
    print(f"  GT Labels (padded mit -1) shape: {dummy_gt_labels_b.shape}, Beispiel Sample 0: {dummy_gt_labels_b[0].tolist()}")
    print(f"  GT Boxes (gepadded) shape: {dummy_gt_boxes_b.shape}")
    print(f"  GT Valid Mask shape (Info): {dummy_gt_valid_mask_b.shape}, Beispiel Sample 0: {dummy_gt_valid_mask_b[0].tolist()}")

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
    )
    
    print("\nTeste HungarianMatcher separat:")
    matched_indices = matcher_instance(dummy_pred_logits, dummy_pred_boxes, dummy_gt_labels_b, dummy_gt_boxes_b)
    for i_sample, (pred_idx, gt_idx) in enumerate(matched_indices):
        print(f"  Sample {i_sample}: Matched Pred Indices: {pred_idx.tolist()}, Matched GT Indices (relativ zu validen GTs): {gt_idx.tolist()}")
        num_valid_gt_sample_i = (dummy_gt_labels_b[i_sample] >=0).sum().item()
        if gt_idx.numel() > 0 and num_valid_gt_sample_i > 0:
            if not (gt_idx.max().item() < num_valid_gt_sample_i) :
                 print(f"    WARNUNG: GT Index {gt_idx.max().item()} könnte außerhalb der Grenzen für {num_valid_gt_sample_i} valide GTs in Sample {i_sample} liegen.")
        elif gt_idx.numel() > 0 and num_valid_gt_sample_i == 0:
            print(f"    WARNUNG: Matches gefunden ({gt_idx.tolist()}), aber keine validen GTs in Sample {i_sample} laut Label-Padding.")

    print("\nTeste SetCriterion.forward():")
    calculated_losses = criterion(
        decoder_outputs={"pred_logits": dummy_pred_logits, "pred_boxes": dummy_pred_boxes},
        gt_labels_b=dummy_gt_labels_b, 
        gt_boxes_b=dummy_gt_boxes_b    
    )

    print("\nBerechnete Verluste (ungewichtet):")
    for loss_name, loss_value in calculated_losses.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")
    
    total_weighted_loss = torch.tensor(0.0)
    if calculated_losses: 
        for k_loss in calculated_losses.keys():
            if k_loss in loss_weight_dict_cfg: 
                total_weighted_loss += calculated_losses[k_loss] * loss_weight_dict_cfg[k_loss]
    print(f"  Gewichteter Gesamtverlust: {total_weighted_loss.item():.4f}")
    
    print("\nSetCriterion and HungarianMatcher example run successful.")