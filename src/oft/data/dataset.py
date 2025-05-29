#!/usr/bin/env python3
import numpy as np
from torch.utils.data import Dataset
from typing import List, Optional, Dict, Any
import traceback 

from oft.utils.common_utils import parse_scene_description
from oft.utils.config import load_config 
from truckscenes import TruckScenes 
from truckscenes.utils.data_classes import Box as DevkitBox

class TruckScenesDataset(Dataset):
    def __init__(
        self,
        dataroot: str,
        version: str,
        history_window: int = 1, 
        max_boxes: Optional[int] = None, 
        augment_noise_std: float = 0.0,
    ):
        self.ts = TruckScenes(version=version, dataroot=dataroot)
        self.history_window_for_dataset_item = history_window 
        self.max_boxes = max_boxes
        self.augment_noise_std = augment_noise_std

        self.samples: List[str] = []
        for scene in self.ts.scene:
            tok = scene["first_sample_token"]
            while tok:
                self.samples.append(tok)
                sample_rec = self.ts.get("sample", tok)
                if not sample_rec: 
                    break
                tok = sample_rec["next"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        token = self.samples[idx]
        sample = self.ts.get("sample", token)
        if not sample:
            raise IndexError(f"Sample-Token {token} nicht im Devkit gefunden.")

        ts_current = sample["timestamp"]
        prev_token   = sample["prev"]
        next_token   = sample["next"]
        scene_token  = sample["scene_token"]
        ann_tokens   = sample.get("anns", []) 

        # --- Lade Metadaten (wie gehabt) ---
        sample_data: Dict[str, Any]  = {}
        calib_rec: Dict[str, Any]    = {}
        ego_pose_rec: Dict[str, Any] = {}
        for ch, sd_tok in sample.get("data", {}).items():
            sd = self.ts.get("sample_data", sd_tok)
            if not sd: continue
            sample_data[ch]  = sd
            cs_tok = sd.get("calibrated_sensor_token")
            ep_tok = sd.get("ego_pose_token")
            if cs_tok: calib_rec[ch] = self.ts.get("calibrated_sensor", cs_tok)
            if ep_tok: ego_pose_rec[ch] = self.ts.get("ego_pose", ep_tok)

        ann_rec_dict: Dict[str, Any] = {} # Umbenannt für Klarheit
        inst_rec_dict: Dict[str, Any] = {} 
        attr_rec_dict: Dict[str, Any] = {}
        vis_rec_dict: Dict[str, Any] = {}

        for a_tok in ann_tokens:
            ann = self.ts.get("sample_annotation", a_tok)
            if not ann: 
                continue
            ann_rec_dict[a_tok] = ann
            inst_tok = ann.get("instance_token")
            if inst_tok:
                 instance = self.ts.get("instance", inst_tok)
                 if instance: inst_rec_dict[inst_tok] = instance 
            attr_tokens_for_ann = ann.get("attribute_tokens", [])
            attr_rec_dict[a_tok] = [self.ts.get("attribute", at) for at in attr_tokens_for_ann if self.ts.get("attribute", at)]
            vis_tok = ann.get("visibility_token")
            if vis_tok:
                visibility = self.ts.get("visibility", vis_tok)
                if visibility : vis_rec_dict[a_tok] = visibility
        
        scene_rec   = self.ts.get("scene", scene_token)
        scene_meta  = parse_scene_description(scene_rec["description"]) if scene_rec else {}

        # --- Boxen für aktuellen Frame laden UND Geschwindigkeiten berechnen ---
        current_boxes_list = []
        velocities_list = []
        
        print(f"DEBUG Dataset: Sample {token}, Anzahl ann_tokens: {len(ann_tokens)}")

        for ann_token_current in ann_tokens:
            try:
                box_current = self.ts.get_box(ann_token_current)
                if box_current is None:
                    print(f"  DEBUG Dataset: Box für ann_token {ann_token_current} ist None.")
                    continue

                cx, cy, cz = box_current.center
                w, l, h    = box_current.wlh 
                orient     = box_current.orientation
                if orient is None or not hasattr(orient, 'yaw_pitch_roll'):
                    print(f"  DEBUG Dataset: Orientierungsproblem für ann_token {ann_token_current}.")
                    continue 
                yaw = orient.yaw_pitch_roll[0]
                current_boxes_list.append(np.array([cx, cy, cz, w, l, h, yaw], dtype=np.float32))

                # Geschwindigkeitsberechnung basierend auf prev_ann_token
                velocity_for_this_box = np.zeros(3, dtype=np.float32) # Default
                
                # Hole den aktuellen sample_annotation record
                current_sample_ann_record = ann_rec_dict.get(ann_token_current)
                if current_sample_ann_record:
                    prev_ann_token = current_sample_ann_record.get("prev")
                    if prev_ann_token: # Wenn es eine vorherige Annotation für diese Instanz gibt
                        prev_sample_ann_record = self.ts.get("sample_annotation", prev_ann_token)
                        if prev_sample_ann_record:
                            # Holen des sample_tokens und Timestamps der vorherigen Annotation
                            prev_sample_token_for_ann = prev_sample_ann_record.get("sample_token")
                            prev_sample_record = self.ts.get("sample", prev_sample_token_for_ann)
                            
                            if prev_sample_record:
                                ts_prev_ann = prev_sample_record["timestamp"]
                                dt = (ts_current - ts_prev_ann) * 1e-6 # in Sekunden

                                if dt > 1e-6: # Sinnvoller Zeitunterschied
                                    box_prev = self.ts.get_box(prev_ann_token)
                                    if box_prev and hasattr(box_prev, 'center'):
                                        displacement = box_current.center - box_prev.center
                                        velocity_for_this_box = displacement / dt
                                        print(f"  DEBUG VEL: Sample {token}, ann {ann_token_current}: Vel berechnet: {velocity_for_this_box.tolist()}, dt={dt:.4f}")
                                    else:
                                        print(f"  DEBUG VEL: Sample {token}, ann {ann_token_current}: Konnte prev_box nicht laden für Vel-Berechnung.")
                                else:
                                    print(f"  DEBUG VEL: Sample {token}, ann {ann_token_current}: dt ({dt:.4f}s) zu klein für Vel-Berechnung.")
                            else:
                                print(f"  DEBUG VEL: Sample {token}, ann {ann_token_current}: Konnte prev_sample_record nicht laden für Vel-Berechnung.")
                        else:
                            print(f"  DEBUG VEL: Sample {token}, ann {ann_token_current}: Konnte prev_sample_ann_record nicht laden für Vel-Berechnung.")
                    else:
                        print(f"  DEBUG VEL: Sample {token}, ann {ann_token_current}: Kein prev_ann_token vorhanden.")
                else:
                    print(f"  DEBUG VEL: Sample {token}, ann {ann_token_current}: Kein current_sample_ann_record gefunden.")
                
                velocities_list.append(velocity_for_this_box)

            except Exception as e:
                print(f"WARNUNG Dataset __getitem__: Fehler beim Holen/Verarbeiten von Box/Velocity für Annotation '{ann_token_current}' in Sample '{token}': {type(e).__name__} - {e}")
                traceback.print_exc()
                # Wenn ein Fehler bei einer Box auftritt, fügen wir auch keine Geschwindigkeit hinzu, um Konsistenz zu wahren
                if len(current_boxes_list) > len(velocities_list):
                    current_boxes_list.pop() # Entferne die zuletzt hinzugefügte Box, wenn Velocity nicht hinzugefügt werden konnte
                pass 
        
        current = np.vstack(current_boxes_list) if current_boxes_list else np.zeros((0,7), dtype=np.float32)
        velocities = np.vstack(velocities_list) if velocities_list else np.zeros((0,3), dtype=np.float32)

        # Sicherstellen, dass current und velocities die gleiche Anzahl an Zeilen haben
        if current.shape[0] != velocities.shape[0]:
            print(f"WARNUNG Dataset __getitem__: Inkonsistente Anzahl Boxen ({current.shape[0]}) und Velocities ({velocities.shape[0]}) für Sample {token} nach Verarbeitung. Setze Velocities auf Nullen.")
            velocities = np.zeros((current.shape[0], 3), dtype=np.float32)
        
        print(f"DEBUG Dataset __getitem__: Sample {token}, Final current shape: {current.shape}, Final velocities shape: {velocities.shape}")
        if velocities.shape[0] > 0:
            print(f"  DEBUG VEL: Erste berechnete Velocity für Sample {token}: {velocities[0].tolist()}")


        # --- History-Frames laden (Logik bleibt ähnlich, aber ohne Geschwindigkeitsberechnung hier) ---
        item_history: List[Optional[np.ndarray]] = [] 
        # history_frame_timestamps ist hier nicht mehr primär für Velocity nötig, aber kann für andere Zwecke bleiben
        
        prev_hist_frame_token_for_history_array = sample["prev"] # Eigener Iterator für History-Array
        for _ in range(self.history_window_for_dataset_item): 
            if prev_hist_frame_token_for_history_array:
                s_hist = self.ts.get("sample", prev_hist_frame_token_for_history_array)
                if not s_hist: break 
                
                arrs_h = []
                hist_ann_tokens = s_hist.get("anns", [])
                for a_hist_tok in hist_ann_tokens:
                    try:
                        box_h = self.ts.get_box(a_hist_tok)
                        if box_h is None: continue
                        cx_h, cy_h, cz_h = box_h.center
                        w_h, l_h, h_h    = box_h.wlh
                        orient_h         = box_h.orientation
                        if orient_h is None or not (hasattr(orient_h, 'yaw_pitch_roll')): continue
                        yaw_h = orient_h.yaw_pitch_roll[0]
                        arrs_h.append(np.array([cx_h, cy_h, cz_h, w_h, l_h, h_h, yaw_h], dtype=np.float32))
                    except Exception: 
                        pass
                item_history.append(np.vstack(arrs_h) if arrs_h else np.zeros((0,7), dtype=np.float32))
                prev_hist_frame_token_for_history_array = s_hist["prev"]
            else:
                item_history.append(None)
        
        padding_mask = np.array(
            [0 if (h is None or h.shape[0] == 0) else 1 for h in item_history] + [1 if current.shape[0] > 0 else 0],
            dtype=np.uint8
        )
        
        if self.augment_noise_std > 0 and current.shape[0] > 0:
            current[:, :3] += np.random.normal(
                loc=0.0, scale=self.augment_noise_std, size=(current.shape[0], 3)
            )

        if self.max_boxes is not None and current.shape[0] > self.max_boxes:
            print(f"WARNUNG Dataset: Zu viele Boxen ({current.shape[0]}) für Sample {token}, max_boxes={self.max_boxes}. Schneide ab.")
            current = current[:self.max_boxes, :]
            if velocities.shape[0] > self.max_boxes: 
                velocities = velocities[:self.max_boxes, :]
        
        print(f"DEBUG Dataset __getitem__: Sample {token} - Returning {current.shape[0]} boxes, {velocities.shape[0]} velocities.")

        return {
            "sample_token":     token, "prev": prev_token, "next": next_token,
            "scene_token":      scene_token, "timestamp": ts_current,
            "sample_data":      sample_data, "calibrated_sensor": calib_rec, "ego_pose": ego_pose_rec,
            "anns":             ann_tokens, # Liste der Annotation-Tokens
            "annotation":       ann_rec_dict, # Dict der Annotation-Records
            "instance":         inst_rec_dict, # Dict der Instanz-Records
            "attributes":       attr_rec_dict, 
            "visibility":       vis_rec_dict,
            "scene_meta":       scene_meta,
            "current":          current,
            "history":          item_history, 
            "padding_mask":     padding_mask, 
            "velocities":       velocities, # Jetzt mit GT-verknüpften Geschwindigkeiten
        }