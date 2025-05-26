#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
import random # For dropout
from pathlib import Path # Hinzugefügt

# Interne Projekt-Imports
from oft.utils.config import load_config
from oft.data.dataset import TruckScenesDataset # Deine Dataset-Klasse
from oft.utils.common_utils import _to_json_serializable # Dein JSON-Serialisierungshelfer

def apply_noise_to_boxes(boxes_world_7d: np.ndarray, noise_std_actual: float) -> np.ndarray:
    """
    Applies Gaussian noise to the centers (x,y,z) of the given boxes.
    Args:
        boxes_world_7d (np.ndarray): Array of shape (N,7) [x,y,z,w,l,h,yaw] in world coords.
        noise_std_actual (float): Single float value for std dev of noise applied to x,y,z.
                                 If 0, no noise is applied.
    Returns:
        np.ndarray: Noisy boxes, same shape as input.
    """
    if noise_std_actual <= 0.0 or boxes_world_7d.shape[0] == 0:
        return boxes_world_7d.copy()

    noisy_boxes = boxes_world_7d.copy()
    noise_vectors = np.random.normal(
        loc=0.0,
        scale=noise_std_actual, 
        size=(noisy_boxes.shape[0], 3) 
    )
    noisy_boxes[:, :3] += noise_vectors
    return noisy_boxes

def apply_dropout_and_get_indices(all_boxes_7d: np.ndarray, dropout_rate: float):
    """
    Randomly removes a fraction of boxes and returns the kept boxes and their original indices.
    Args:
        all_boxes_7d (np.ndarray): Array of shape (N,7).
        dropout_rate (float): Fraction of boxes to remove (0.0 to 1.0).
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Array of boxes after dropout, shape (M,7) where M <= N.
            - Array of original indices of the kept boxes, shape (M,).
    """
    if not (0.0 <= dropout_rate <= 1.0):
        raise ValueError(f"Dropout rate muss zwischen 0.0 und 1.0 liegen, ist aber {dropout_rate}")
    
    n_total_boxes = all_boxes_7d.shape[0]
    if dropout_rate == 0.0 or n_total_boxes == 0:
        return all_boxes_7d.copy(), np.arange(n_total_boxes, dtype=int) # Stelle sicher, dass Indizes Integer sind
    
    num_to_keep = round(n_total_boxes * (1.0 - dropout_rate))

    if num_to_keep <= 0:
        return np.zeros((0,7), dtype=all_boxes_7d.dtype), np.array([], dtype=int)

    all_indices = np.arange(n_total_boxes, dtype=int)
    np.random.shuffle(all_indices) 
    
    # Behalte die ersten num_to_keep Indizes nach dem Mischen
    keep_indices_shuffled = all_indices[:int(num_to_keep)]
    # Sortiere die Indizes, um die ursprüngliche relative Reihenfolge der beibehaltenen Boxen zu erhalten.
    # Das ist wichtig, um sie korrekt mit Geschwindigkeiten und Kategorien zu matchen.
    keep_indices_sorted = np.sort(keep_indices_shuffled)
    
    kept_boxes = all_boxes_7d[keep_indices_sorted].copy()
    
    return kept_boxes, keep_indices_sorted


def main():
    parser = argparse.ArgumentParser(description="A1: Generate SEQUENTIAL Simulated Sensor Detections from GT")
    parser.add_argument("-c", "--pipeline", required=False, default="config/pipeline.yaml",
                        help="Pfad zur pipeline.yaml.")
    args = parser.parse_args()

    cfg = load_config(args.pipeline) # Lädt die Konfiguration
    dcfg = cfg["dataset"]
    vcfg = cfg["visualization"]
    ocfg = cfg["output"]
    sim_cfg = cfg.get("simulation", {})

    if not sim_cfg.get("enabled", False):
        print("INFO A1: Simulation ist in der pipeline.yaml deaktiviert. Überspringe Generierung.")
        return

    target_sample_idx = int(vcfg.get("sample_idx", 0))
    history_window = int(vcfg.get("history_window", 0))
    
    base_noise_std_from_config = float(dcfg.get("augment_noise_std", 0.0)) 

    print(f"INFO A1: Initialisiere TruckScenesDataset für sequenzielle Detektionsgenerierung...")
    # Das history_window im Dataset-Konstruktor ist für die item["history"] und item["velocities"] Berechnung.
    # Es ist nicht direkt dasselbe wie das history_window für die Sequenzlänge hier, aber es ist gut,
    # dass das Dataset Geschwindigkeiten berechnen kann, wenn history_window > 0 ist.
    ds_clean = TruckScenesDataset(
        dataroot=str(dcfg["dataroot"]),
        version=str(dcfg["version"]).strip(),
        history_window=int(dcfg.get("history_window_for_velocity_calc", 1)), # Expliziter Config-Wert oder Default für Dataset-interne History
        max_boxes=int(dcfg.get("gt_max_boxes") or 0) if dcfg.get("gt_max_boxes") is not None else 0, # Stellt sicher, dass es ein Int ist
        augment_noise_std=0.0 # Wir laden saubere GT; Rauschen wird hier im Skript pro Fake-Sensor gesteuert
    )

    if not (0 <= target_sample_idx < len(ds_clean.samples)):
        print(f"FEHLER A1: visualization.sample_idx ({target_sample_idx}) ist außerhalb des Dataset-Bereichs (0-{len(ds_clean.samples)-1}).")
        print(f"FEHLER A1: Bitte passe visualization.sample_idx in deiner '{args.pipeline}' an.")
        return
    
    # --- Sequenz von Sample-Tokens bestimmen ---
    sample_tokens_in_sequence = []
    # Starte mit dem Token des Ziel-Frames (letzter Frame der Sequenz)
    current_token_for_seq_build = ds_clean.samples[target_sample_idx] 

    for _ in range(history_window + 1): # +1 um den target_sample_idx selbst einzuschließen
        sample_tokens_in_sequence.append(current_token_for_seq_build)
        sample_record = ds_clean.ts.get("sample", current_token_for_seq_build) # Hole den Sample-Record
        prev_token = sample_record["prev"] # Hole den Token des vorherigen Samples
        if not prev_token: # Wenn es keinen vorherigen Frame gibt (Anfang der Szene)
            break
        current_token_for_seq_build = prev_token
    sample_tokens_in_sequence.reverse() # Ordne die Tokens vom ältesten zum neuesten Frame

    if not sample_tokens_in_sequence:
        print("FEHLER A1: Keine Sample-Tokens für die Sequenz gefunden.")
        return

    print(f"INFO A1: Generiere simulierte Detektionen für {len(sample_tokens_in_sequence)} Frames, endend mit Index {target_sample_idx} (Token: {sample_tokens_in_sequence[-1]})")
    print(f"INFO A1: Verwendete Sample-Tokens in der Sequenz: {sample_tokens_in_sequence}")

    # --- Basis-Ausgabeverzeichnis für sequenzielle Daten ---
    # Verwende Path für robustere Pfadoperationen
    general_output_dir = Path(ocfg.get("output_dir", "output")) # Fallback auf "output", falls nicht in config
    sequential_sim_base_dir = general_output_dir / "simulated_detections_sequence"
    sequential_sim_base_dir.mkdir(parents=True, exist_ok=True) # Erstelle Verzeichnis, falls nicht vorhanden
    print(f"INFO A1: Ausgaben werden in '{sequential_sim_base_dir.resolve()}' gespeichert.")

    simulated_sources_config = sim_cfg.get("sources", {})
    if not simulated_sources_config:
        print("WARNUNG A1: Keine 'simulation.sources' in der pipeline.yaml definiert. Keine Daten generiert.")
        return

    # --- Äußere Schleife: Durchlaufe jeden Frame in der Sequenz ---
    for current_frame_token in sample_tokens_in_sequence:
        print(f"\n  Verarbeite Frame (Sample-Token): {current_frame_token}")
        
        try:
            # Hole das aufbereitete Item (GT-Boxen, Geschwindigkeiten etc.) für den aktuellen Frame-Token
            # Finde den Index des aktuellen Tokens in der Sample-Liste des Datasets
            idx_for_current_frame = ds_clean.samples.index(current_frame_token) 
            item_gt = ds_clean[idx_for_current_frame] # Rufe __getitem__ des Datasets auf
        except ValueError:
            print(f"FEHLER A1: Sample-Token {current_frame_token} nicht in ds_clean.samples gefunden. Überspringe Frame.")
            continue
            
        # Hole saubere GT-Boxen und Geschwindigkeiten aus dem Dataset-Item
        gt_boxes_world_clean_7d = item_gt["current"].copy() # (N,7) [x,y,z,w,l,h,yaw]
        gt_velocities_world_3d = item_gt["velocities"].copy() # (N,3) [vx,vy,vz]
        
        # Kategorienamen extrahieren
        category_names_for_current_frame = []
        # Stelle sicher, dass 'anns' existiert und die Länge mit der Anzahl der Boxen übereinstimmt
        num_gt_boxes = gt_boxes_world_clean_7d.shape[0]
        annotation_tokens = item_gt.get("anns", [])

        if num_gt_boxes > 0:
            if len(annotation_tokens) == num_gt_boxes:
                for ann_token in annotation_tokens:
                    try:
                        # Greife auf die von TruckScenesDataset geladenen Records zu
                        annotation_record = item_gt["annotation"].get(ann_token)
                        if not annotation_record:
                            print(f"WARNUNG A1: Kein Annotation-Record für ann_token {ann_token} in Frame {current_frame_token} gefunden (via item_gt['annotation']).")
                            category_names_for_current_frame.append("unknown")
                            continue
                        
                        instance_record = item_gt["instance"].get(ann_token) # Annahme: key ist ann_token, nicht instance_token
                        if not instance_record: # Korrekte Abfrage, falls instance_token der Key ist
                            instance_token_from_ann = annotation_record.get("instance_token")
                            instance_record = item_gt["instance"].get(instance_token_from_ann)
                            if not instance_record:
                                print(f"WARNUNG A1: Kein Instanz-Record für ann_token {ann_token} (Instanz: {instance_token_from_ann}) in Frame {current_frame_token} gefunden.")
                                category_names_for_current_frame.append("unknown")
                                continue
                        
                        category_token = instance_record.get("category_token")
                        if not category_token:
                            print(f"WARNUNG A1: Kein Category-Token im Instanz-Record für ann_token {ann_token} in Frame {current_frame_token} gefunden.")
                            category_names_for_current_frame.append("unknown")
                            continue

                        category_record = ds_clean.ts.get("category", category_token) # Direkter Zugriff auf Devkit für Category
                        if not category_record or "name" not in category_record:
                            print(f"WARNUNG A1: Kein Kategorie-Record oder Name für category_token {category_token} (ann_token {ann_token}) in Frame {current_frame_token} gefunden.")
                            category_names_for_current_frame.append("unknown")
                            continue
                        category_names_for_current_frame.append(category_record["name"])
                    except Exception as e:
                        print(f"WARNUNG A1: Fehler beim Extrahieren des Kategorienamens für ann_token {ann_token} in Frame {current_frame_token}: {e}")
                        category_names_for_current_frame.append("unknown") # Fallback
            else:
                print(f"WARNUNG A1: Anzahl der Annotation-Tokens ({len(annotation_tokens)}) stimmt nicht mit Anzahl der GT-Boxen ({num_gt_boxes}) für Frame {current_frame_token} überein. Kategorien werden als 'unknown' gesetzt.")
                category_names_for_current_frame = ["unknown"] * num_gt_boxes
        
        if num_gt_boxes == 0:
            print(f"    INFO A1: Keine GT-Boxen im Frame {current_frame_token} gefunden. Es werden leere JSON-Dateien für diesen Frame erzeugt.")
        else:
            print(f"    Geladene saubere GT für Token {current_frame_token}: {num_gt_boxes} Boxen.")

        # --- Innere Schleife: Durchlaufe jeden konfigurierten Fake-Sensor ---
        for sensor_name, sensor_params in simulated_sources_config.items():
            # `output_file` aus der Config wird hier ignoriert, da wir [sample_token].json verwenden
            print(f"    Generiere Daten für simulierten Sensor: {sensor_name}")
            
            sensor_specific_output_dir = sequential_sim_base_dir / sensor_name
            sensor_specific_output_dir.mkdir(parents=True, exist_ok=True)
            # Dateiname ist jetzt der Sample-Token des aktuellen Frames
            output_filename_for_frame = f"{current_frame_token}.json" 
            output_path = sensor_specific_output_dir / output_filename_for_frame

            # Starte immer mit den sauberen GT-Daten für diesen Frame und Sensor
            current_boxes_to_process_7d = gt_boxes_world_clean_7d.copy()
            current_velocities_to_process_3d = gt_velocities_world_3d.copy()
            current_categories_to_process = list(category_names_for_current_frame) # Kopie der Liste

            # 1. Rauschen auf Positionen anwenden
            noise_multiplier = float(sensor_params.get("noise_std_multiplier", 1.0))
            actual_noise_to_apply = base_noise_std_from_config * noise_multiplier
            if actual_noise_to_apply > 0 and current_boxes_to_process_7d.shape[0] > 0:
                current_boxes_to_process_7d = apply_noise_to_boxes(current_boxes_to_process_7d, actual_noise_to_apply)
            # Hier könnte man optional Rauschen auf Geschwindigkeiten hinzufügen, falls konfiguriert

            # 2. Dropout anwenden (und Indizes für Geschwindigkeiten/Kategorien bekommen)
            dropout_rate = float(sensor_params.get("dropout_rate", 0.0))
            # Standardmäßig alle Indizes behalten, falls kein Dropout oder keine Boxen
            kept_indices = np.arange(current_boxes_to_process_7d.shape[0], dtype=int) 
            
            if dropout_rate > 0 and current_boxes_to_process_7d.shape[0] > 0:
                current_boxes_to_process_7d, kept_indices = apply_dropout_and_get_indices(current_boxes_to_process_7d, dropout_rate)
            
            # Wende Dropout auch auf Geschwindigkeiten und Kategorien an, basierend auf den `kept_indices`
            if current_boxes_to_process_7d.shape[0] > 0 : 
                if kept_indices.size > 0 : # Nur wenn Indizes vorhanden sind
                    # Stelle sicher, dass current_velocities_to_process_3d und current_categories_to_process die richtige Länge haben
                    if current_velocities_to_process_3d.shape[0] == len(kept_indices) or current_velocities_to_process_3d.shape[0] == num_gt_boxes: # num_gt_boxes ist die Länge vor dem Dropout
                         current_velocities_to_process_3d = current_velocities_to_process_3d[kept_indices]
                    else: # Fallback, falls Längen nicht passen (sollte nicht passieren, wenn kept_indices korrekt sind)
                        print(f"      WARNUNG A1: Längen-Mismatch bei Geschwindigkeits-Dropout für Sensor {sensor_name}, Frame {current_frame_token}. Velocities werden ggf. nicht korrekt gedroptout.")
                        # Behalte alle Velocities oder setze sie auf Null, je nach gewünschtem Verhalten
                        # current_velocities_to_process_3d = current_velocities_to_process_3d # Behalte alle
                        # Oder: current_velocities_to_process_3d = np.zeros((current_boxes_to_process_7d.shape[0], 3))

                    if len(current_categories_to_process) == len(kept_indices) or len(current_categories_to_process) == num_gt_boxes:
                        current_categories_to_process = [current_categories_to_process[i] for i in kept_indices]
                    else:
                        print(f"      WARNUNG A1: Längen-Mismatch bei Kategorie-Dropout für Sensor {sensor_name}, Frame {current_frame_token}. Kategorien werden ggf. nicht korrekt gedroptout.")
                        # current_categories_to_process = current_categories_to_process # Behalte alle
                        # Oder: current_categories_to_process = ["unknown"] * current_boxes_to_process_7d.shape[0]

                else: # Alle Boxen wurden gedroptout, leere auch Velocities und Kategorien
                    current_velocities_to_process_3d = np.zeros((0,3), dtype=gt_velocities_world_3d.dtype)
                    current_categories_to_process = []
            elif current_boxes_to_process_7d.shape[0] == 0: # Keine Boxen nach Rauschen oder von Anfang an
                 current_velocities_to_process_3d = np.zeros((0,3), dtype=gt_velocities_world_3d.dtype)
                 current_categories_to_process = []

            # 3. In JSON-Format konvertieren
            simulated_score = float(sensor_params.get("score", 0.8))
            detections_for_json_list = []
            if current_boxes_to_process_7d.shape[0] > 0:
                # Stelle sicher, dass die Anzahl der Velocities und Kategorien mit den Boxen übereinstimmt
                if not (current_boxes_to_process_7d.shape[0] == current_velocities_to_process_3d.shape[0] == len(current_categories_to_process)):
                    print(f"      WARNUNG A1: Inkonsistente Anzahl von Boxen ({current_boxes_to_process_7d.shape[0]}), "
                          f"Velocities ({current_velocities_to_process_3d.shape[0]}), "
                          f"und Kategorien ({len(current_categories_to_process)}) "
                          f"für Sensor {sensor_name}, Frame {current_frame_token} vor JSON-Erstellung. Überspringe Detektionserstellung für diesen Sensor/Frame.")
                else:
                    for i in range(current_boxes_to_process_7d.shape[0]):
                        box_7d = current_boxes_to_process_7d[i]
                        vel_3d = current_velocities_to_process_3d[i]
                        cat_name = current_categories_to_process[i]

                        detections_for_json_list.append({
                            "sample_token": current_frame_token, 
                            "translation_world": box_7d[0:3].tolist(),
                            "size_wlh": box_7d[3:6].tolist(), 
                            "rotation_yaw_world": float(box_7d[6]),
                            "velocity_world": vel_3d.tolist(), 
                            "detection_name": cat_name,
                            "detection_score": simulated_score 
                        })
            
            # JSON-Struktur für die Ausgabe pro Frame und Sensor
            frame_output_json_content = {
                "meta": {
                    "sensor_modality": sensor_params.get("modality", "simulated"), # Hole Modalität aus Sensor-Params
                    "sensor_name": sensor_name,
                    "source_sample_token": current_frame_token, # Token des Original-Frames
                    "cfg_noise_multiplier": noise_multiplier,
                    "cfg_dropout_rate": dropout_rate,
                    "cfg_simulated_score": simulated_score
                },
                "results": {
                    # Der Key ist der Sample-Token des Frames, zu dem die Detektionen gehören
                    current_frame_token: detections_for_json_list 
                }
            }
            
            with open(output_path, "w") as f:
                json.dump(frame_output_json_content, f, indent=2, default=_to_json_serializable)
            # print(f"      Gespeichert: {len(detections_for_json_list)} Detektionen in {output_path}")

    print(f"\n✓ A1: Generierung sequenzieller simulierter Detektionen abgeschlossen.")

if __name__ == "__main__":
    main()
