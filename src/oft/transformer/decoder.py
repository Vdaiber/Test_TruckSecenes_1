# src/oft/transformer/decoder.py
import torch
import torch.nn as nn
import math # Hinzugefügt für math.pi
from typing import Optional, Dict, List

# Importe für das Laden der Konfiguration im Test-Block
import os
from oft.utils.config import load_config # Stellen Sie sicher, dass dieser Importpfad korrekt ist
import numpy as np # Für den Testblock

class ObjectFusionTransformerDecoder(nn.Module):
    """
    Transformer-Decoder zur Vorhersage einer Menge von Objekten basierend auf
    Encoder-Memory und lernbaren Objekt-Queries.
    Gibt Boxen mit tatsächlichen Dimensionen und Boxen mit log-Dimensionen für den Verlust zurück.
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
                 box_dim: int = 7 # Erwartet immer 7: cx, cy, cz, w, l, h, yaw (bzw. log(w),log(l),log(h) vom MLP)
                ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        # box_dim ist die Dimension des Outputs des bbox_head_mlp,
        # der cx,cy,cz, log(w),log(l),log(h), yaw_raw vorhersagt.

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
        
        # Bbox_head_mlp: Sagt cx, cy, cz, log(w), log(l), log(h), raw_yaw vorher
        self.bbox_head_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, box_dim) # box_dim ist 7
        )

    def forward(self,
                memory: torch.Tensor, 
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
            - "pred_boxes_for_loss": Box-Parameter für den Verlust
                                     (B, NumQueries, 7) -> (cx,cy,cz, log(w),log(l),log(h), yaw_proc).
            - "pred_boxes_for_matching_and_giou": Box-Parameter mit realen Dimensionen
                                     (B, NumQueries, 7) -> (cx,cy,cz, w,l,h, yaw_proc).
        """
        batch_size = memory.shape[0]
        # tgt sind die lernbaren Objekt-Queries
        tgt = self.object_queries_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1) # (B, NumQueries, d_model)

        # Decoder-Durchlauf
        decoder_output = self.transformer_decoder_stack(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=None, 
            memory_key_padding_mask=memory_key_padding_mask
        ) # (B, NumQueries, d_model)
         
        # Klassifikations-Vorhersagen
        pred_logits = self.class_head(decoder_output) # (B, NumQueries, NumClasses + 1)
         
        # Rohe Bounding-Box-Vorhersagen vom MLP
        # Erwartet: (cx, cy, cz, log(w), log(l), log(h), raw_yaw)
        raw_pred_box_params = self.bbox_head_mlp(decoder_output) # (B, NumQueries, 7)
         
        # Extrahieren und verarbeiten der Parameter
        pred_cxcycz = raw_pred_box_params[..., :3]  # Absolute Fahrzeugkoordinaten-Zentren
        pred_log_wlh = raw_pred_box_params[..., 3:6] # Logarithmierte Dimensionen
        
        # Yaw verarbeiten (tanh skaliert auf [-1, 1], dann * pi auf [-pi, pi])
        pred_yaw_proc = torch.tanh(raw_pred_box_params[..., 6:7]) * math.pi # (B, NumQueries, 1)
        
        # Boxen für den Verlust (mit log-Dimensionen)
        pred_boxes_for_loss = torch.cat((pred_cxcycz, pred_log_wlh, pred_yaw_proc), dim=-1)
        
        # Boxen mit realen Dimensionen (für Matching, GIoU, Inferenz)
        pred_wlh_actual = torch.exp(pred_log_wlh) # Exponentieren der log-Dimensionen
        pred_boxes_for_matching_and_giou = torch.cat((pred_cxcycz, pred_wlh_actual, pred_yaw_proc), dim=-1)
         
        return {
            "pred_logits": pred_logits,
            "pred_boxes_for_loss": pred_boxes_for_loss,
            "pred_boxes_for_matching_and_giou": pred_boxes_for_matching_and_giou
        }

if __name__ == '__main__':
    print("Running ObjectFusionTransformerDecoder example with config-loaded parameters (Log-Dims)...")

    config_file_path_dec = "config/pipeline_c_modules.yaml"
    project_root_dec = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    config_file_path_dec_abs = os.path.join(project_root_dec, config_file_path_dec)

    if not os.path.exists(config_file_path_dec_abs):
        print(f"WARNUNG: Config file not found at {config_file_path_dec_abs}. Using default test parameters.")
        model_cfg_dec = {}
    else:
        try:
            full_pipeline_config_dec = load_config(config_file_path_dec_abs)
            print(f"Konfiguration für Decoder-Test geladen von: {config_file_path_dec_abs}")
            model_cfg_dec = OmegaConf.to_container(full_pipeline_config_dec.get('model', {}), resolve=True)
        except Exception as e:
            print(f"Fehler beim Laden der Konfiguration für Decoder-Test: {e}. Verwende Standard-Fallback-Parameter.")
            model_cfg_dec = {}

    d_model_cfg = model_cfg_dec.get('d_model', 256)
    n_heads_cfg = model_cfg_dec.get('nhead', 8)
    num_dec_layers_cfg = model_cfg_dec.get('num_decoder_layers', 3)
    dim_ff_decoder_cfg = model_cfg_dec.get('dim_feedforward_decoder', d_model_cfg * 4)
    dropout_cfg = model_cfg_dec.get('dropout', 0.1)
    activation_cfg = model_cfg_dec.get('activation', 'relu')
    num_queries_cfg = model_cfg_dec.get('num_queries', 100)
    num_classes_cfg = model_cfg_dec.get('num_classes', 12) # Angepasst an deine Config
    box_dim_cfg = model_cfg_dec.get('box_dim', 7)
     
    batch_s = 2
    num_enc_objects = 50 
     
    dummy_encoder_memory = torch.rand(batch_s, num_enc_objects, d_model_cfg)
    dummy_memory_padding_mask = torch.zeros(batch_s, num_enc_objects, dtype=torch.bool)
    if num_enc_objects > 10:
        dummy_memory_padding_mask[:, -10:] = True

    print(f"\nVerwendete Decoder-Parameter:")
    print(f"  d_model: {d_model_cfg}, nhead: {n_heads_cfg}, num_decoder_layers: {num_dec_layers_cfg}")
    print(f"  dim_feedforward: {dim_ff_decoder_cfg}, dropout: {dropout_cfg}, activation: {activation_cfg}")
    print(f"  num_queries: {num_queries_cfg}, num_classes (ohne BG): {num_classes_cfg-1}, box_dim_mlp_output: {box_dim_cfg}")

    decoder = ObjectFusionTransformerDecoder(
        d_model=d_model_cfg, nhead=n_heads_cfg, num_decoder_layers=num_dec_layers_cfg, 
        dim_feedforward=dim_ff_decoder_cfg, dropout=dropout_cfg, activation=activation_cfg,
        num_queries=num_queries_cfg, num_classes=num_classes_cfg, box_dim=box_dim_cfg
    )
    decoder.eval()

    with torch.no_grad():
        predictions = decoder(
            memory=dummy_encoder_memory,
            memory_key_padding_mask=dummy_memory_padding_mask
        )

    print(f"\nDecoder Output Dictionary Keys: {list(predictions.keys())}")
    print(f"  pred_logits shape: {predictions['pred_logits'].shape}") 
    print(f"  pred_boxes_for_loss shape: {predictions['pred_boxes_for_loss'].shape}")   
    print(f"  pred_boxes_for_matching_and_giou shape: {predictions['pred_boxes_for_matching_and_giou'].shape}")   

    assert predictions['pred_logits'].shape == (batch_s, num_queries_cfg, num_classes_cfg + 1)
    assert predictions['pred_boxes_for_loss'].shape == (batch_s, num_queries_cfg, box_dim_cfg)
    assert predictions['pred_boxes_for_matching_and_giou'].shape == (batch_s, num_queries_cfg, box_dim_cfg)
     
    print(f"\nBeispiel Output für erste Query, erstes Sample:")
    pred_loss_box_sample = predictions['pred_boxes_for_loss'][0, 0, :].cpu().numpy()
    pred_match_box_sample = predictions['pred_boxes_for_matching_and_giou'][0, 0, :].cpu().numpy()

    print(f"  Logits (erste 5 Werte): {predictions['pred_logits'][0, 0, :5].tolist()}")
    print(f"  Box für Loss (cx,cy,cz, log(w),log(l),log(h), yaw): {np.round(pred_loss_box_sample, 3)}")
    print(f"  Box für Matching/GIoU (cx,cy,cz, w,l,h, yaw): {np.round(pred_match_box_sample, 3)}")
    print(f"    cx,cy,cz (gleich): {np.round(pred_match_box_sample[:3], 3)}")
    print(f"    w,l,h (exponentiert): {np.round(pred_match_box_sample[3:6], 3)}")
    print(f"    yaw (gleich, nach tanh*pi): {pred_match_box_sample[6]:.3f} rad")
     
    print("\nObjectFusionTransformerDecoder example run successful (mit Log-Dimensionen).")
