# src/oft/transformer/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional, Tuple
import numpy as np

# Placeholder für eine BEV IoU-Berechnung in PyTorch.
# Die Implementierung aus fusion/nms_3d.py (Shapely-basiert) ist nicht direkt für den Loss geeignet.
# Man würde hier eine PyTorch-basierte Variante benötigen oder auf L1/GIoU für 3D ausweichen.
def placeholder_bev_iou_loss(boxes1_7d: torch.Tensor, boxes2_7d: torch.Tensor) -> torch.Tensor:
    """
    Platzhalter für eine BEV IoU-Berechnung (oder GIoU) für 7D-Boxen (cx,cy,cz,w,l,h,yaw).
    Sollte einen IoU-Wert pro Boxenpaar zurückgeben. 1 - IoU wäre dann der Loss.
    Für eine echte Implementierung müsste man die Boxen in BEV-Polygone umwandeln und deren IoU berechnen.
    """
    # Einfacher L1-Loss auf den ersten 2 Dimensionen (XY-Zentrum) als grober Platzhalter
    # Dies ist KEIN IoU-Loss, nur um die Struktur zu zeigen!
    if boxes1_7d.numel() == 0 or boxes2_7d.numel() == 0:
        return torch.tensor(0.0, device=boxes1_7d.device)
    
    # Nehmen wir an, boxes1 ist (N, 7) und boxes2 ist (M, 7)
    # Wir wollen eine (N, M) Matrix von Verlusten.
    # Für diesen Platzhalter: L1-Distanz der XY-Zentren. Höher ist schlechter.
    # Skaliere, um grob im Bereich [0,1] zu sein für Kosten.
    cost = torch.cdist(boxes1_7d[:, :2], boxes2_7d[:, :2], p=1) / 10.0 # Normierungsfaktor 10 ist willkürlich
    return torch.clamp(cost, max=1.0) # Begrenze auf 1 für Kosten


class HungarianMatcher(nn.Module):
    """
    Führt das optimale bipartite Matching zwischen Vorhersagen und Ground Truth durch.
    Verwendet den Ungarischen Algorithmus.
    """
    def __init__(self,
                 cost_class: float = 1.0,      # Gewicht für Klassifikationskosten
                 cost_bbox_l1: float = 1.0,    # Gewicht für L1-Box-Regressionskosten
                 cost_giou_bev: float = 1.0):  # Gewicht für GIoU/BEV-IoU-Box-Kosten
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox_l1 = cost_bbox_l1
        self.cost_giou_bev = cost_giou_bev
        if cost_class == 0 and cost_bbox_l1 == 0 and cost_giou_bev == 0:
            raise ValueError("Alle Kosten-Gewichte im Matcher dürfen nicht null sein.")

    @torch.no_grad() # Wichtig: Matching erfordert keine Gradienten
    def forward(self,
                pred_logits: torch.Tensor,  # (Batch, NumQueries, NumClasses + 1)
                pred_boxes: torch.Tensor,   # (Batch, NumQueries, BoxDim=7)
                gt_labels: torch.Tensor,    # (Batch, NumGTObjects) LongTensor
                gt_boxes: torch.Tensor      # (Batch, NumGTObjects, BoxDim=7)
               ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Führt das Matching für einen Batch durch.

        Args:
            pred_logits: Logits der Klassifikationsvorhersagen.
            pred_boxes: Vorhergesagte Box-Parameter.
            gt_labels: Ground-Truth Klassenlabels.
            gt_boxes: Ground-Truth Box-Parameter.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: Eine Liste von Tupeln für jedes Batch-Element.
                Jedes Tupel enthält (matched_pred_indices, matched_gt_indices).
        """
        batch_size, num_queries = pred_logits.shape[:2]
        
        # Wir berechnen die Kostenmatrix für jedes Batch-Element einzeln
        indices = []
        for i in range(batch_size):
            # Aktuelle Vorhersagen und GTs für dieses Batch-Element
            # pred_logits_i: (NumQueries, NumClasses + 1)
            # pred_boxes_i: (NumQueries, BoxDim)
            # gt_labels_i: (NumGT_i)
            # gt_boxes_i: (NumGT_i, BoxDim)
            
            # Filtere Padding aus GTs für dieses Sample, falls gt_labels Padding-Werte enthält (z.B. -1)
            # Die Collate-Funktion liefert gt_target_valid_mask, die hier nützlich wäre.
            # Annahme: gt_labels enthält keine Padding-Werte mehr oder wurde bereits gefiltert.
            # Wenn gt_labels -1 für Padding enthält:
            valid_gt_mask_i = (gt_labels[i] >= 0)
            if not valid_gt_mask_i.any(): # Keine validen GTs für dieses Sample
                indices.append((torch.tensor([], dtype=torch.long, device=pred_logits.device),
                                torch.tensor([], dtype=torch.long, device=pred_logits.device)))
                continue

            gt_labels_i = gt_labels[i][valid_gt_mask_i]
            gt_boxes_i = gt_boxes[i][valid_gt_mask_i]
            num_gt_i = gt_labels_i.shape[0]

            if num_gt_i == 0: # Keine GT-Objekte nach Filterung
                indices.append((torch.tensor([], dtype=torch.long, device=pred_logits.device),
                                torch.tensor([], dtype=torch.long, device=pred_logits.device)))
                continue

            pred_logits_i = pred_logits[i] # (NumQueries, NumClasses + 1)
            pred_boxes_i = pred_boxes[i]   # (NumQueries, BoxDim)

            # 1. Klassifikationskosten
            # Kosten sind -P(gt_class | pred_query). Niedriger ist besser.
            # (NumQueries, NumClasses + 1) -> (NumQueries, NumGT_i)
            prob = pred_logits_i.softmax(-1) 
            # Nimm die Wahrscheinlichkeiten für die tatsächlichen GT-Klassen
            # gt_labels_i ist (NumGT_i), prob ist (NumQueries, NumClasses+1)
            # Wir brauchen prob[:, gt_labels_i] -> (NumQueries, NumGT_i)
            cost_class_matrix = -prob[:, gt_labels_i]

            # 2. Box-Regressionskosten (L1)
            # (NumQueries, BoxDim) und (NumGT_i, BoxDim) -> (NumQueries, NumGT_i)
            # cdist berechnet paarweise Distanzen.
            # Für Box-Parameter (x,y,z,w,l,h,yaw) ist L1 sinnvoll.
            # Yaw-Differenz muss ggf. speziell behandelt werden (z.B. abs(atan2(sin,cos) - atan2(sin,cos)))
            # Hier vereinfacht: L1 auf alle 7 Parameter.
            cost_bbox_l1_matrix = torch.cdist(pred_boxes_i, gt_boxes_i, p=1)
            # Normalisierung der Yaw-Differenz könnte hier wichtig sein, wenn yaw direkt verwendet wird.

            # 3. Box-Regressionskosten (GIoU / BEV IoU) - Platzhalter
            # (NumQueries, NumGT_i)
            # Dies erfordert eine PyTorch-basierte IoU-Berechnung.
            cost_giou_matrix = placeholder_bev_iou_loss(pred_boxes_i, gt_boxes_i) # Höher ist schlechter

            # 4. Gesamtkostenmatrix
            # (NumQueries, NumGT_i)
            C = (self.cost_class * cost_class_matrix +
                 self.cost_bbox_l1 * cost_bbox_l1_matrix +
                 self.cost_giou_bev * cost_giou_matrix)
            
            # Konvertiere zu CPU NumPy für linear_sum_assignment
            C_np = C.detach().cpu().numpy()
            
            # Ungarischer Algorithmus
            # row_ind sind Indizes der Queries, col_ind sind Indizes der GTs
            row_ind, col_ind = linear_sum_assignment(C_np)
            
            indices.append((torch.as_tensor(row_ind, dtype=torch.long, device=pred_logits.device),
                            torch.as_tensor(col_ind, dtype=torch.long, device=pred_logits.device)))
        return indices


class SetCriterion(nn.Module):
    """
    Diese Klasse berechnet den Verlust für ein Set-Prediction-Modell.
    Der Prozess beinhaltet:
    1. Matching der Vorhersagen mit den Ground-Truth-Objekten (HungarianMatcher).
    2. Berechnung der Verluste (Klassifikation, Box-Regression) für die gematchten Paare.
    """
    def __init__(self, 
                 num_classes: int, # Anzahl der Objektklassen (ohne Hintergrund)
                 matcher: HungarianMatcher,
                 eos_coef: float, # Relatives Gewicht für die "kein Objekt" Klasse im Klassifikationsloss
                 losses: List[str], # Liste der zu berechnenden Verluste, z.B. ['labels', 'boxes_l1', 'boxes_giou']
                 weight_dict: Dict[str, float]): # Gewichte für die einzelnen Verlustkomponenten
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef # Gewicht für die "no-object" Klasse
        self.losses = losses
        self.weight_dict = weight_dict

        # Definiere die "kein Objekt" Klasse als self.num_classes
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef # Letzte Klasse ist "kein Objekt"
        self.register_buffer('empty_weight', empty_weight)

    def _get_src_permutation_idx(self, indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Hilfsfunktion, um Batch- und Query-Indizes für gematchte Vorhersagen zu erhalten. """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Hilfsfunktion, um Batch- und GT-Indizes für gematchte GTs zu erhalten. """
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_labels(self, 
                    pred_logits: torch.Tensor, # (B, NumQueries, NumClasses + 1)
                    gt_labels_b: torch.Tensor,   # (B, NumGTObjects)
                    indices: List[Tuple[torch.Tensor, torch.Tensor]], 
                    num_total_boxes: int):
        """
        Klassifikationsloss (z.B. CrossEntropy).
        Für nicht gematchte Queries ist das Ziel die "kein Objekt"-Klasse.
        """
        # pred_logits: (Batch, NumQueries, NumClasses + 1)
        # gt_labels_b: (Batch, NumGTObjects) - enthält Klassenlabels für GTs in diesem Batch
        # indices: Liste von (matched_pred_indices, matched_gt_indices) für jedes Batch-Element

        # Erzeuge Ziel-Labels für alle Queries
        # Shape: (BatchSize, NumQueries)
        # Initialisiere mit "kein Objekt"-Klasse
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes,
                                    dtype=torch.long, device=pred_logits.device)
        
        # Setze die korrekten Klassen für gematchte Queries
        # matched_pred_indices_batch: (TotalMatchedPredsInBatch) - enthält Query-Indizes
        # matched_gt_indices_batch: (TotalMatchedPredsInBatch) - enthält GT-Indizes (relativ zu GTs im Sample)
        # batch_indices_for_preds: (TotalMatchedPredsInBatch) - Batch-Index für jede gematchte Vorhersage
        batch_indices_for_preds, matched_pred_indices_batch = self._get_src_permutation_idx(indices)
        
        # Wir brauchen die GT-Labels für die gematchten GTs
        # gt_labels_b ist (Batch, NumGTObjects)
        # Wir müssen die Labels für die spezifischen matched_gt_indices pro Batch-Element holen.
        
        # Sammle die GT-Labels für die gematchten Paare
        target_classes_o = [] 
        for i, (_, matched_gt_idx_sample) in enumerate(indices):
            # gt_labels_b[i] ist (NumGT_i)
            # matched_gt_idx_sample ist (NumMatched_i)
            if matched_gt_idx_sample.numel() > 0: # Nur wenn es Matches in diesem Sample gibt
                 target_classes_o.append(gt_labels_b[i][matched_gt_idx_sample])
        
        if target_classes_o: # Nur wenn es überhaupt Matches im Batch gab
            target_classes_o = torch.cat(target_classes_o)
            target_classes[batch_indices_for_preds, matched_pred_indices_batch] = target_classes_o
        
        # Berechne CrossEntropyLoss
        # pred_logits.transpose(1,2) -> (Batch, NumClasses+1, NumQueries) für CrossEntropy
        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, self.empty_weight)
        
        losses = {'loss_ce': loss_ce}
        # Optional: Accuracy berechnen (hier nicht implementiert)
        return losses

    def loss_boxes_l1(self, 
                      pred_boxes: torch.Tensor, # (B, NumQueries, BoxDim)
                      gt_boxes_b: torch.Tensor,   # (B, NumGTObjects, BoxDim)
                      indices: List[Tuple[torch.Tensor, torch.Tensor]],
                      num_total_boxes: int):
        """ L1-Loss für Box-Regression. """
        # Extrahiere die gematchten Vorhersagen und GTs
        batch_indices_for_preds, matched_pred_indices_batch = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[batch_indices_for_preds, matched_pred_indices_batch] # (TotalMatched, BoxDim)

        target_boxes_list = []
        for i, (_, matched_gt_idx_sample) in enumerate(indices):
            if matched_gt_idx_sample.numel() > 0:
                target_boxes_list.append(gt_boxes_b[i][matched_gt_idx_sample])
        
        if not target_boxes_list : # Keine Matches im gesamten Batch
            return {'loss_bbox_l1': torch.tensor(0.0, device=pred_boxes.device)}

        target_boxes = torch.cat(target_boxes_list, dim=0) # (TotalMatched, BoxDim)

        loss_bbox_l1 = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        # Summiere über Box-Dimensionen und dann über alle gematchten Boxen, normalisiere mit Anzahl Boxen
        losses['loss_bbox_l1'] = loss_bbox_l1.sum() / num_total_boxes 
        return losses

    def loss_boxes_giou(self, 
                        pred_boxes: torch.Tensor, 
                        gt_boxes_b: torch.Tensor,
                        indices: List[Tuple[torch.Tensor, torch.Tensor]],
                        num_total_boxes: int):
        """ GIoU-Loss für Box-Regression (Platzhalter). """
        # Extrahiere die gematchten Vorhersagen und GTs
        batch_indices_for_preds, matched_pred_indices_batch = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[batch_indices_for_preds, matched_pred_indices_batch]

        target_boxes_list = []
        for i, (_, matched_gt_idx_sample) in enumerate(indices):
            if matched_gt_idx_sample.numel() > 0:
                target_boxes_list.append(gt_boxes_b[i][matched_gt_idx_sample])
        
        if not target_boxes_list:
            return {'loss_giou': torch.tensor(0.0, device=pred_boxes.device)}
            
        target_boxes = torch.cat(target_boxes_list, dim=0)

        # Hier würde die echte GIoU-Berechnung erfolgen.
        # placeholder_bev_iou_loss gibt Kosten zurück (höher ist schlechter).
        # GIoU Loss ist typischerweise 1 - GIoU.
        # Für diesen Platzhalter nehmen wir an, dass die Funktion direkt einen Loss-Wert liefert.
        # Wir müssen es paarweise für src_boxes und target_boxes berechnen und dann mitteln.
        # Da placeholder_bev_iou_loss eine Kostenmatrix (N,M) erwartet, ist es hier nicht direkt anwendbar.
        # Wir verwenden stattdessen einen Dummy-L1-Loss als Platzhalter für GIoU.
        
        # --- ECHTE GIoU BERECHNUNG WÄRE HIER NÖTIG ---
        # Beispiel: loss_giou = (1 - generalized_iou(src_boxes_bev, target_boxes_bev)).diag()
        # Wobei generalized_iou die paarweise GIoU zwischen den gematchten Boxen berechnet.
        # Und src_boxes_bev / target_boxes_bev wären die BEV-Repräsentationen.
        
        # Als Platzhalter: L1-Loss auf XY-Positionen und WL-Dimensionen
        # Dies ist KEIN GIoU-Loss!
        loss_placeholder_giou = F.l1_loss(src_boxes[:, [0,1,3,4]], target_boxes[:, [0,1,3,4]], reduction='none')
        
        losses = {}
        losses['loss_giou'] = loss_placeholder_giou.sum() / num_total_boxes
        return losses


    def forward(self, 
                decoder_outputs: Dict[str, torch.Tensor], # Enthält 'pred_logits' und 'pred_boxes'
                gt_valid_mask_b: torch.Tensor, # (Batch, NumGTObjects) - True für valide GTs
                gt_labels_b: torch.Tensor,   # (Batch, NumGTObjects)
                gt_boxes_b: torch.Tensor      # (Batch, NumGTObjects, BoxDim)
               ) -> Dict[str, torch.Tensor]:
        """
        Berechnet den Gesamtverlust.
        Args:
            decoder_outputs: Dictionary mit 'pred_logits' und 'pred_boxes' vom Decoder.
            gt_valid_mask_b: Maske, die anzeigt, welche GT-Objekte valide sind (nicht Padding).
            gt_labels_b: Ground-Truth Klassenlabels.
            gt_boxes_b: Ground-Truth Box-Parameter.
        Returns:
            Ein Dictionary mit allen berechneten Verlustkomponenten.
        """
        pred_logits = decoder_outputs['pred_logits']
        pred_boxes = decoder_outputs['pred_boxes']

        # Führe das Matching für den gesamten Batch durch
        # Wichtig: gt_labels_b und gt_boxes_b könnten Padding enthalten, wenn die Anzahl der GTs
        # pro Sample im Batch unterschiedlich ist und die Collate-Fn sie auf eine max. Länge paddet.
        # Der Matcher muss mit dieser Situation umgehen oder die gepaddeten GTs müssen vorher
        # herausgefiltert werden. Die `gt_valid_mask_b` aus der Collate-Funktion ist hierfür ideal.
        
        # Bereite GTs für Matcher vor: Filtere Padding-GTs pro Batch-Element
        # Der Matcher erwartet, dass gt_labels und gt_boxes nur die validen GTs enthalten.
        # Wir müssen eine Liste von Tensoren für gt_labels und gt_boxes erstellen.
        
        # Die `HungarianMatcher`-Implementierung oben filtert bereits intern basierend auf gt_labels >= 0.
        # Wir können also die gepaddeten gt_labels_b und gt_boxes_b direkt übergeben.
        # Die `gt_valid_mask_b` wird hier nicht direkt an den Matcher übergeben, aber
        # die Collate-Fn sollte sicherstellen, dass gt_labels für Padding z.B. -1 ist.
        
        indices = self.matcher(pred_logits, pred_boxes, gt_labels_b, gt_boxes_b)

        # Berechne die Anzahl der tatsächlich gematchten Boxen (für Normalisierung der Verluste)
        num_total_matched_boxes = sum(len(t[0]) for t in indices)
        num_total_matched_boxes = torch.as_tensor([num_total_matched_boxes], dtype=torch.float, device=pred_logits.device)
        
        # Für verteilte Szenarien (nicht relevant für uns im Moment)
        # if dist.is_available() and dist.is_initialized():
        #     torch.distributed.all_reduce(num_total_boxes)
        # num_total_boxes = torch.clamp(num_total_boxes / dist.get_world_size(), min=1).item()
        num_total_matched_boxes = torch.clamp(num_total_matched_boxes, min=1).item()


        # Berechne alle angeforderten Verluste
        losses = {}
        for loss_type in self.losses:
            if loss_type == 'labels':
                losses.update(self.loss_labels(pred_logits, gt_labels_b, indices, num_total_matched_boxes))
            elif loss_type == 'boxes_l1':
                losses.update(self.loss_boxes_l1(pred_boxes, gt_boxes_b, indices, num_total_matched_boxes))
            elif loss_type == 'boxes_giou': # Oder 'boxes_bev_iou'
                losses.update(self.loss_boxes_giou(pred_boxes, gt_boxes_b, indices, num_total_matched_boxes))
            else:
                raise ValueError(f"Unbekannter Verlusttyp: {loss_type}")
        
        return losses

if __name__ == '__main__':
    print("Running SetCriterion and HungarianMatcher example...")

    # Beispiel-Parameter
    bs = 2
    num_queries = 10
    num_classes_no_bg = 5 # 5 echte Klassen
    box_d = 7
    
    # Dummy Decoder Outputs
    dummy_pred_logits = torch.rand(bs, num_queries, num_classes_no_bg + 1)
    dummy_pred_boxes = torch.rand(bs, num_queries, box_d) * 50 # Boxen in Weltkoordinaten

    # Dummy GT Daten (aus Collate-Funktion)
    # Annahme: max_gt_objects im Batch ist 4
    max_gt_objs = 4
    dummy_gt_labels = torch.randint(0, num_classes_no_bg, (bs, max_gt_objs), dtype=torch.long)
    dummy_gt_boxes = torch.rand(bs, max_gt_objs, box_d) * 50
    
    # Beispiel: Sample 0 hat 3 GTs, Sample 1 hat 2 GTs
    # Setze Padding-Labels auf -1 (oder einen anderen Wert, den der Matcher/Loss ignoriert)
    dummy_gt_labels[0, 3:] = -1 
    dummy_gt_labels[1, 2:] = -1
    
    # Die `gt_valid_mask` wäre hier:
    # [[True, True, True, False], [True, True, False, False]]

    print(f"\nDummy Inputs:")
    print(f"  Pred Logits: {dummy_pred_logits.shape}")
    print(f"  Pred Boxes: {dummy_pred_boxes.shape}")
    print(f"  GT Labels (mit -1 für Padding): {dummy_gt_labels}")
    print(f"  GT Boxes: {dummy_gt_boxes.shape}")

    # Initialisiere Matcher und Criterion
    matcher_instance = HungarianMatcher(cost_class=1.0, cost_bbox_l1=5.0, cost_giou_bev=2.0)
    
    # Gewichte für die Verlustkomponenten
    loss_weights = {'loss_ce': 1.0, 'loss_bbox_l1': 5.0, 'loss_giou': 2.0}
    
    criterion = SetCriterion(num_classes=num_classes_no_bg, 
                             matcher=matcher_instance, 
                             eos_coef=0.1, # Gewicht für "kein Objekt" Klasse
                             losses=['labels', 'boxes_l1', 'boxes_giou'],
                             weight_dict=loss_weights)
    
    # Berechne Verluste
    # Die `gt_valid_mask` wird aktuell nicht direkt an `criterion.forward` übergeben,
    # da der Matcher und die Loss-Funktionen mit den gepaddeten GTs umgehen (durch -1 Label).
    # Man könnte sie aber übergeben, um expliziter zu sein.
    calculated_losses = criterion(
        decoder_outputs={"pred_logits": dummy_pred_logits, "pred_boxes": dummy_pred_boxes},
        gt_valid_mask_b= (dummy_gt_labels >= 0), # Erzeuge valid mask für die Ausgabe
        gt_labels_b=dummy_gt_labels,
        gt_boxes_b=dummy_gt_boxes
    )

    print("\nCalculated Losses:")
    for loss_name, loss_value in calculated_losses.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")
    
    # Berechne den Gesamtverlust (gewichtete Summe)
    total_loss = sum(calculated_losses[k] * loss_weights[k] for k in calculated_losses.keys() if k in loss_weights)
    print(f"  Total Weighted Loss: {total_loss.item():.4f}")
    
    print("\nSetCriterion and HungarianMatcher example run successful.")