# src/oft/data/dataset.py

import numpy as np
from torch.utils.data import Dataset
from typing import List, Optional, Dict

from oft.utils.common_utils import parse_scene_description
from oft.utils.config import load_config # Bereits importiert, sehr gut!
from truckscenes import TruckScenes

class TruckScenesDataset(Dataset):
    """
    Lädt per TruckScenes-Devkit Samples und ergänzt:
      - alle Sample-Infos (prev/next, timestamp, scene_token)
      - alle sample_data-Records + calib + ego_pose pro Kanal
      - alle Annotation-Records + Instanz-, Attribut- und Visibility-Infos
      - current & history arrays (N×7: x,y,z,w,l,h,yaw)
      - velocities (N×3) berechnet aus erstem History-Frame
      - padding_mask (H+1,)
      - scene_meta (parsed description)
    """
    def __init__(
        self,
        dataroot: str,
        version: str,
        history_window: int = 5,
        max_boxes: int = 50,
        augment_noise_std: float = 0.0, # Dieser Wert kommt aus der Initialisierung, typischerweise von einem Skript
    ):
        self.ts = TruckScenes(version=version, dataroot=dataroot)
        self.history_window = history_window
        self.max_boxes = max_boxes
        # Wichtig: self.augment_noise_std wird bei der Erzeugung des Dataset-Objekts gesetzt.
        # Normalerweise lesen Skripte (wie stage2 oder stage3) den Wert aus der pipeline.yaml
        # und übergeben ihn hierher.
        self.augment_noise_std = augment_noise_std

        # chronologische Liste aller sample_tokens
        self.samples: List[str] = []
        for scene in self.ts.scene:
            tok = scene["first_sample_token"]
            while tok:
                self.samples.append(tok)
                tok = self.ts.get("sample", tok)["next"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        # cfg = load_config() # cfg wird hier schon geladen, das ist gut
        # In deiner __getitem__ wird load_config() bereits aufgerufen.
        # Das bedeutet, wir können auf die globale Konfiguration zugreifen.
        # Wichtig: `load_config()` ohne Argument lädt die Standard-Config. Wenn du
        # von einem Skript eine spezifische Config-Datei per -c Option übergibst,
        # muss `load_config(pfad_zur_config)` aufgerufen werden.
        # Deine Skripte scheinen das aber korrekt zu machen, indem sie `load_config(args.pipeline)`
        # am Anfang aufrufen und dann `cfg` herumreichen.
        # Der Aufruf von `load_config()` hier im Dataset ist also für den Fall,
        # dass das Dataset *ohne* explizit übergebene `cfg` von außen genutzt wird.
        # Für unseren Fall ist es besser, wenn die Skripte (stage2, stage3)
        # den Wert aus der `cfg` lesen und an __init__ übergeben.
        # Deine __init__ nimmt `augment_noise_std` bereits entgegen. Das ist der richtige Weg.

        # Dein Code lädt die config hier:
        cfg = load_config() # Dies lädt die Standard-Config, typischerweise `config/pipeline.yaml`

        token = self.samples[idx]
        sample = self.ts.get("sample", token)

        # --- Sample-level meta ---
        prev_token   = sample["prev"]
        next_token   = sample["next"]
        scene_token  = sample["scene_token"]
        timestamp    = sample["timestamp"]
        ann_tokens   = sample["anns"]  # Liste der Annotation-Tokens

        # --- sample_data + calib + ego_pose pro Kanal ---
        sample_data  = {}
        calib_rec    = {}
        ego_pose_rec = {}
        for ch, sd_tok in sample["data"].items():
            sd = self.ts.get("sample_data", sd_tok)
            sample_data[ch]  = sd
            calib_rec[ch]    = self.ts.get("calibrated_sensor", sd["calibrated_sensor_token"])
            ego_pose_rec[ch] = self.ts.get("ego_pose",        sd["ego_pose_token"])

        # --- Annotation-Level Infos ---
        ann_rec     = {a: self.ts.get("sample_annotation", a) for a in ann_tokens}
        inst_rec    = {a: self.ts.get("instance",            ann_rec[a]["instance_token"])
                       for a in ann_tokens}
        attr_rec    = {a: [self.ts.get("attribute", at) for at in ann_rec[a]["attribute_tokens"]]
                       for a in ann_tokens}
        vis_rec     = {a: self.ts.get("visibility", ann_rec[a]["visibility_token"])
                       for a in ann_tokens if ann_rec[a]["visibility_token"]}

        # --- Szene-Meta (geparst) ---
        scene_rec   = self.ts.get("scene", scene_token)
        scene_meta  = parse_scene_description(scene_rec["description"])

        # --- 1) Current-Boxes extrahieren (N×7) ---
        arrs = []
        for a in ann_tokens:
            box = self.ts.get_box(a) # Liefert Box in Weltkoordinaten
            cx, cy, cz = box.center
            w, l, h    = box.wlh
            orient     = box.orientation
            # yaw extrahieren
            if hasattr(orient, "yaw_pitch_roll"):
                yaw = orient.yaw_pitch_roll[0]
            elif hasattr(orient, "angle"): # Für 2D Boxen oder falls Winkel direkt als yaw gegeben ist
                yaw = orient.angle
            else: # Fallback, falls Orientierung nur als Skalar (z.B. direkter Winkel)
                yaw = float(orient)
            arrs.append(np.array([cx, cy, cz, w, l, h, yaw], dtype=np.float32))
        current = np.vstack(arrs) if arrs else np.zeros((0,7), dtype=np.float32)

        # --- 2) History-Frames laden ---
        history: List[Optional[np.ndarray]] = []
        prev_hist_tok = sample["prev"] # Korrigiert von prev_tok zu prev_hist_tok für Klarheit
        for _ in range(self.history_window):
            if prev_hist_tok:
                s_hist = self.ts.get("sample", prev_hist_tok) # Korrigiert von s zu s_hist
                arrs_h = []
                for a_hist in s_hist["anns"]: # Korrigiert von a zu a_hist
                    box_h = self.ts.get_box(a_hist) # Korrigiert von box zu box_h
                    cx_h, cy_h, cz_h = box_h.center
                    w_h, l_h, h_h    = box_h.wlh
                    orient_h         = box_h.orientation
                    if hasattr(orient_h, "yaw_pitch_roll"):
                        yaw_h = orient_h.yaw_pitch_roll[0]
                    elif hasattr(orient_h, "angle"):
                        yaw_h = orient_h.angle
                    else:
                        yaw_h = float(orient_h)
                    arrs_h.append(np.array([cx_h, cy_h, cz_h, w_h, l_h, h_h, yaw_h], dtype=np.float32))
                history.append(np.vstack(arrs_h) if arrs_h else np.zeros((0,7), dtype=np.float32))
                prev_hist_tok = s_hist["prev"]
            else:
                history.append(None) # Fügt None hinzu, wenn keine weiteren History-Frames vorhanden sind

        # --- 3) Padding-Maske (0=padd, 1=real) ---
        mask = [
            0 if (h is None or (isinstance(h, np.ndarray) and h.shape[0] == 0)) else 1
            for h in history
        ] + [1]  # current immer valid
        padding_mask = np.array(mask, dtype=np.uint8)  # Form: (history_window+1,)

        # --- 4) Gaussian-Noise auf current-Box-Zentren (optional) ---
        # HIER IST DIE STELLE, DIE WIR ANPASSEN
        # `self.augment_noise_std` wird bei der Initialisierung des Datasets gesetzt.
        # Die Skripte (stage2, stage3) lesen `dataset.augment_noise_std` aus der config
        # und übergeben es an den Konstruktor des Datasets.
        # Jetzt müssen wir zusätzlich den Schalter `tracking.use_noisy_gt_as_input` prüfen,
        # aber *nur wenn diese Funktion im Kontext von Stage 3 aufgerufen wird*.
        # Das ist das knifflige.
        #
        # Einfachere Lösung für jetzt:
        # Stage 2 (visualize_noise) wird `augment_noise_std` im Konstruktor setzen (z.B. auf den Wert aus der config).
        # Stage 3 (track_history) wird `augment_noise_std` im Konstruktor setzen, *abhängig* vom
        # `tracking.use_noisy_gt_as_input` Schalter in der config.
        # Das bedeutet, die Logik, ob Rauschen angewendet wird oder nicht,
        # wird von dem Skript gesteuert, das das `TruckScenesDataset` Objekt erstellt.
        # Die `__getitem__` Methode hier unten wendet dann Rauschen an, wenn `self.augment_noise_std > 0` ist.
        # Das ist bereits so implementiert und das ist gut!

        # Deine aktuelle Implementierung für Rauschen:
        if self.augment_noise_std > 0 and current.shape[0] > 0:
            current[:, :3] += np.random.normal( # Hier wird direkt auf current addiert
                loc=0.0,
                scale=self.augment_noise_std, # self.augment_noise_std wird im Konstruktor gesetzt
                size=(current.shape[0], 3)
            )

        # --- 5) Geschwindigkeiten aus erstem History-Frame (optional) ---
        # Beachte: Wenn 'current' verrauscht wurde, beeinflusst das auch die Geschwindigkeitsberechnung hier.
        if history and history[0] is not None and current.shape[0] > 0 and history[0].shape[0] > 0:
            # Um sicherzustellen, dass die Geschwindigkeitsberechnung robust ist,
            # falls die Anzahl der Boxen nicht übereinstimmt (z.B. durch Filterung oder Fehler),
            # könnte man hier ein Matching oder eine andere Logik benötigen.
            # Für den Moment gehen wir von einer 1:1 Korrespondenz aus, wenn die Shapes passen.
            # Eine robustere Variante wäre, Boxen über IDs zu matchen, falls verfügbar.
            # Da IDs hier nicht direkt verwendet werden, ist die Annahme bei gleicher Anzahl okay für den Start.
            
            # Einfache Differenz, wenn Anzahl der Boxen gleich ist
            if current.shape[0] == history[0].shape[0]:
                 velocities = current[:, :3] - history[0][:, :3] # (N,3)
            else:
                # Fallback, wenn die Anzahl der Boxen nicht übereinstimmt
                # logger.warning(f"Shape mismatch for velocity calc: current {current.shape[0]}, history[0] {history[0].shape[0]}. Velocities set to None.")
                velocities = np.zeros((current.shape[0], 3), dtype=np.float32) # Oder None, oder leeres Array
        else:
            velocities = np.zeros((current.shape[0], 3), dtype=np.float32) # Fallback zu Null-Geschwindigkeiten

        return {
            # sample-level
            "sample_token":     token,
            "prev":             prev_token,
            "next":             next_token,
            "scene_token":      scene_token,
            "timestamp":        timestamp,

            # sample_data + calib + ego_pose
            "sample_data":      sample_data,
            "calibrated_sensor":calib_rec,
            "ego_pose":         ego_pose_rec,

            # annotation-level
            "anns":             ann_tokens,
            "annotation":       ann_rec,
            "instance":         inst_rec,
            "attributes":       attr_rec,
            "visibility":       vis_rec,

            # parsed scene-meta
            "scene_meta":       scene_meta,

            # detected boxes
            "current":          current,     # (N,7) in Weltkoordinaten
            "history":          history,     # List[Optional[np.ndarray(Ni,7)]] in Weltkoordinaten
            "padding_mask":     padding_mask,# (H+1,)
            "velocities":       velocities,  # (N,3) in Weltkoordinaten (oder Nullen)
        }