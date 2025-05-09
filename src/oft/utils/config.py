# src/oft/utils/config.py

import yaml
from pathlib import Path

def load_config(path: str = None) -> dict:
    """
    Lädt die Pipeline-Config aus 'config/pipeline.yaml' im Arbeitsverzeichnis
    oder aus dem übergebenen Pfad.
    """
    if path:
        cfg_file = Path(path)
    else:
        # Arbeitsverzeichnis (docker WORKDIR /app) → /app/config/pipeline.yaml
        cfg_file = Path.cwd() / "config" / "pipeline.yaml"

    if not cfg_file.exists():
        raise FileNotFoundError(f"Config-Datei nicht gefunden: {cfg_file.resolve()}")

    with cfg_file.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg