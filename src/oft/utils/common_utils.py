# src/oft/utils/common_utils.py

"""
Common utilities for the ObjectFusionTransformer repository.
"""
import json # Hinzugefügt für NumpyEncoder
import numpy as np # Hinzugefügt für NumpyEncoder
from truckscenes import TruckScenes

def _to_json_serializable(obj):
    """
    Konvertiert numpy-Typen in native Python-Typen für JSON-Dump.
    Wird als `default` Argument für `json.dump` oder `json.dumps` verwendet.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Optional: Behandlung für andere Typen wie pyquaternion.Quaternion
    # from pyquaternion import Quaternion
    # if isinstance(obj, Quaternion):
    #     return obj.elements.tolist()  # [w, x, y, z]
    try:
        return str(obj) # Fallback für andere nicht direkt serialisierbare Typen
    except TypeError:
        return repr(obj) # Allerletzter Fallback, gibt eine String-Repräsentation zurück

def list_scenes(dataroot: str, version: str) -> list:
    """
    Liefert die Liste aller Scene-Tokens für das gegebene Dataset.

    Args:
        dataroot: Pfad zum Dataset-Root (z.B. "/data" oder "/data/v1.0-mini")
        version:  Dataset-Version (z.B. "v1.0-mini")

    Returns:
        Eine Liste von Strings mit allen scene["token"]-Werten.
    """
    ts = TruckScenes(version=version, dataroot=dataroot)
    return [scene["token"] for scene in ts.scene]


def parse_scene_description(desc: str) -> dict:
    """
    Parsen des standardisierten description-Strings aus TruckScenes,
    z.B. "weather.clear;daytime.noon;season.autumn;…"
    → { "weather":"clear", "daytime":"noon", "season":"autumn", … }
    """
    md = {}
    for token in desc.split(";"):
        if "." not in token:
            continue
        key, val = token.split(".", 1)
        md[key] = val
    return md

# NEU: Helferklasse für JSON-Serialisierung von NumPy-Typen
# Du kannst sie _to_json_serializable nennen oder NumpyEncoder, wie hier.
# Wenn du sie _to_json_serializable nennen willst, musst du den Import in stage4_fusion.py anpassen.
# Ich nenne sie hier NumpyEncoder, da das gängiger ist.
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        # Optional: Behandlung für andere Typen wie pyquaternion.Quaternion, falls nötig
        # from pyquaternion import Quaternion
        # if isinstance(o, Quaternion):
        #     return o.elements.tolist() # [w, x, y, z]
        return super(NumpyEncoder, self).default(o)

# Die Funktion _to_json_serializable, die dein Stage 3 Skript verwendet,
# ist eine Alternative zur Verwendung der Klasse direkt im json.dump Aufruf.
# Wenn du _to_json_serializable beibehalten willst, definiere sie hier:

