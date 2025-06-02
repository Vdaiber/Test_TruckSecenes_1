# get_signatures.py
import inspect
try:
    from truckscenes.eval.detection.evaluate import DetectionEval
    from truckscenes.eval.detection.config import DetectionConfig

    print("Signatur von DetectionEval.__init__:")
    sig_eval = inspect.signature(DetectionEval.__init__)
    print(sig_eval)

    print("\nSignatur von DetectionConfig.__init__ (für DevKit):")
    sig_config = inspect.signature(DetectionConfig.__init__)
    print(sig_config) # Korrektur: sollte sig_config sein

except ImportError as e:
    print(f"ImportFehler: {e}")
    print("Stelle sicher, dass truckscenes-devkit korrekt installiert ist und im PYTHONPATH liegt.")
except Exception as e:
    print(f"Ein anderer Fehler ist aufgetreten: {e}")
