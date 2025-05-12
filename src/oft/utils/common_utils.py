# objectfusiontransformer/src/oft/utils/common_utils.py

"""
Common utilities for the ObjectFusionTransformer repository.
"""

from truckscenes import TruckScenes

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