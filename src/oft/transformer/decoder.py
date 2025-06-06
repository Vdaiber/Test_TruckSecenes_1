# src/oft/transformer/decoder.py
import torch
import torch.nn as nn
import math
from typing import Optional, Dict, List

# Importe für das Laden der Konfiguration im Test-Block
import os
# from oft.utils.config import load_config # Nur für __main__
from omegaconf import OmegaConf # Nur für __main__
import numpy as np # Nur für __main__

class ObjectFusionTransformerDecoder(nn.Module):
    """
    Transformer-Decoder zur Vorhersage einer Menge von Objekten basierend auf
    Encoder-Memory und lernbaren Objekt-Queries.
    Sagt Box-Parameter als Offsets vorher.
    """
    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 num_queries: int = 100,
                 num_classes: int = 10,
                 box_dim: int = 7, # Sollte 7 sein: cx, cy, cz, log_w, log_l, log_h, yaw
                 center_offset_scale: float = 5.0 # Neuer Parameter für die Skalierung der Zentrum-Offsets
                ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.center_offset_scale = center_offset_scale

        self.object_queries_embed = nn.Embedding(num_queries, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True 
        )
        self.transformer_decoder_stack = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model) 
        )

        self.class_head = nn.Linear(d_model, num_classes + 1) # +1 für "kein Objekt"
        
        # Bbox_head_mlp: Sagt Offsets vorher
        # (Δcx_raw, Δcy_raw, Δcz_raw, Δlog_w, Δlog_l, Δlog_h, Δyaw_raw)
        self.bbox_offset_head_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, box_dim) # box_dim ist 7
        )

    def forward(self,
                memory: torch.Tensor, 
                # Referenz-Boxen werden nicht mehr direkt an den Decoder übergeben,
                # da im GT->GT Fall das Ziel-Offset Null ist und die Rekonstruktion im Loss-Modul erfolgt.
                # Später, mit echten Sensor-Inputs, könnte die Referenz hier oder im Loss explizit benötigt werden.
                memory_key_padding_mask: Optional[torch.Tensor] = None
               ) -> Dict[str, torch.Tensor]:
        """
        Führt den Forward-Pass des Decoders aus.
        Args:
            memory: Output des Encoders (B, NumEncoderObjects, d_model).
            memory_key_padding_mask: Maske für gepaddete Elemente im Encoder-Output.
        Returns:
            Ein Dictionary mit:
            - "pred_logits": Klassifikations-Logits (B, NumQueries, NumClasses + 1).
            - "pred_box_offsets": Vorhergesagte Box-Offsets 
                                   (B, NumQueries, 7) -> (Δcx_scaled, Δcy_scaled, Δcz_scaled, 
                                                          Δlog_w, Δlog_l, Δlog_h, Δyaw_proc).
                                   Zentrum-Offsets sind mit tanh behandelt und skaliert.
        """
        batch_size = memory.shape[0]
        tgt = self.object_queries_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        decoder_output = self.transformer_decoder_stack(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=None, 
            memory_key_padding_mask=memory_key_padding_mask
        )
         
        pred_logits = self.class_head(decoder_output)
         
        # Rohe Offset-Vorhersagen vom MLP
        raw_pred_box_offsets = self.bbox_offset_head_mlp(decoder_output) # (B, NumQueries, 7)
         
        # Zentrum-Offsets mit tanh und Skalierung
        # Diese Offsets sind relativ zu einer impliziten Referenz (z.B. der Query-Position oder (0,0,0))
        # und sollen klein sein.
        pred_center_offsets_normalized = torch.tanh(raw_pred_box_offsets[..., :3]) 
        pred_center_offsets_scaled = pred_center_offsets_normalized * self.center_offset_scale
        
        # Log-Dimensions-Offsets (direkt vom MLP)
        pred_log_dim_offsets = raw_pred_box_offsets[..., 3:6]
        
        # Yaw-Offset (verarbeitet, relativ zu einer Referenz-Yaw von 0)
        pred_yaw_offset_proc = torch.tanh(raw_pred_box_offsets[..., 6:7]) * math.pi
        
        # Kombinierte Offsets für den Verlust
        pred_box_offsets = torch.cat((pred_center_offsets_scaled, 
                                      pred_log_dim_offsets, 
                                      pred_yaw_offset_proc), dim=-1)
         
        return {
            "pred_logits": pred_logits,
            "pred_box_offsets": pred_box_offsets,
            # Die Rekonstruktion zu absoluten Boxen (für GIoU/Matching) wird jetzt im Loss-Modul gemacht,
            # da dort die Referenz (GT-Boxen) verfügbar ist.
        }

if __name__ == '__main__':
    print("Running ObjectFusionTransformerDecoder example with Offset Prediction...")
    # Lade Konfiguration für Testparameter
    # Stelle sicher, dass dieser Pfad relativ zum Ausführungsort des Skripts ist oder absolut.
    # Für Tests direkt aus dem 'transformer' Ordner wäre der Pfad: '../../config/pipeline_c_modules.yaml'
    config_file_path_dec_main = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'pipeline_c_modules.yaml')

    if not os.path.exists(config_file_path_dec_main):
        print(f"WARNUNG: Config file not found at {config_file_path_dec_main}. Using default test parameters.")
        model_cfg_dec_main = { # Minimal-Config für den Test
            'd_model': 256, 'nhead': 8, 'num_decoder_layers': 1, 
            'dim_feedforward_decoder': 512, 'dropout': 0.1, 'activation': 'relu',
            'num_queries': 10, 'num_classes': 12, 'box_dim': 7,
            'center_offset_scale': 5.0 
        }
    else:
        from oft.utils.config import load_config # Importiere hier, da es nur für __main__ benötigt wird
        full_pipeline_config_dec_main = load_config(config_file_path_dec_main)
        model_cfg_dec_main = OmegaConf.to_container(full_pipeline_config_dec_main.get('model', {}), resolve=True)
        # Füge center_offset_scale hinzu, falls nicht in der Config (für Abwärtskompatibilität)
        if 'center_offset_scale' not in model_cfg_dec_main:
            model_cfg_dec_main['center_offset_scale'] = 5.0 
        print(f"Konfiguration für Decoder-Test geladen von: {config_file_path_dec_main}")


    d_model_cfg = model_cfg_dec_main.get('d_model')
    n_heads_cfg = model_cfg_dec_main.get('nhead')
    num_dec_layers_cfg = model_cfg_dec_main.get('num_decoder_layers')
    dim_ff_decoder_cfg = model_cfg_dec_main.get('dim_feedforward_decoder')
    dropout_cfg = model_cfg_dec_main.get('dropout')
    activation_cfg = model_cfg_dec_main.get('activation')
    num_queries_cfg = model_cfg_dec_main.get('num_queries')
    num_classes_cfg = model_cfg_dec_main.get('num_classes')
    box_dim_cfg = model_cfg_dec_main.get('box_dim')
    center_offset_scale_cfg = model_cfg_dec_main.get('center_offset_scale')
     
    batch_s = 2
    num_enc_objects = 50 
    device_test = torch.device("cpu")
     
    dummy_encoder_memory = torch.rand(batch_s, num_enc_objects, d_model_cfg, device=device_test)
    dummy_memory_padding_mask = torch.zeros(batch_s, num_enc_objects, dtype=torch.bool, device=device_test)

    print(f"\nVerwendete Decoder-Parameter:")
    print(f"  d_model: {d_model_cfg}, nhead: {n_heads_cfg}, num_decoder_layers: {num_dec_layers_cfg}")
    print(f"  dim_feedforward: {dim_ff_decoder_cfg}, dropout: {dropout_cfg}, activation: {activation_cfg}")
    print(f"  num_queries: {num_queries_cfg}, num_classes (ohne BG): {num_classes_cfg}")
    print(f"  box_dim (für MLP-Output): {box_dim_cfg}, center_offset_scale: {center_offset_scale_cfg}")

    decoder_instance = ObjectFusionTransformerDecoder(
        d_model=d_model_cfg, nhead=n_heads_cfg, num_decoder_layers=num_dec_layers_cfg, 
        dim_feedforward=dim_ff_decoder_cfg, dropout=dropout_cfg, activation=activation_cfg,
        num_queries=num_queries_cfg, num_classes=num_classes_cfg, box_dim=box_dim_cfg,
        center_offset_scale=center_offset_scale_cfg
    ).to(device_test)
    decoder_instance.eval()

    with torch.no_grad():
        predictions = decoder_instance(
            memory=dummy_encoder_memory,
            memory_key_padding_mask=dummy_memory_padding_mask
        )

    print(f"\nDecoder Output Dictionary Keys: {list(predictions.keys())}")
    print(f"  pred_logits shape: {predictions['pred_logits'].shape}") 
    print(f"  pred_box_offsets shape: {predictions['pred_box_offsets'].shape}")   
    
    assert predictions['pred_logits'].shape == (batch_s, num_queries_cfg, num_classes_cfg + 1)
    assert predictions['pred_box_offsets'].shape == (batch_s, num_queries_cfg, box_dim_cfg)
     
    print(f"\nBeispiel Output für erste Query, erstes Sample:")
    pred_offsets_sample = predictions['pred_box_offsets'][0, 0, :].cpu().numpy()
    print(f"  Logits (erste 5 Werte): {predictions['pred_logits'][0, 0, :5].tolist()}")
    print(f"  Box Offsets (Δcx_s, Δcy_s, Δcz_s, Δlog_w, Δlog_l, Δlog_h, Δyaw_proc): {np.round(pred_offsets_sample, 3)}")
    print(f"    Δcx,y,z (skaliert): {np.round(pred_offsets_sample[:3], 3)}")
    print(f"    Δlog_w,l,h: {np.round(pred_offsets_sample[3:6], 3)}")
    print(f"    Δyaw_proc (nach tanh*pi): {pred_offsets_sample[6]:.3f} rad")
     
    print("\nObjectFusionTransformerDecoder (Offset Prediction) example run successful.")