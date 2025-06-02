import torch
import torch.nn as nn
import math # Hinzugefügt für math.pi
from typing import Optional, Dict, List

# NEU: Importe für das Laden der Konfiguration im Test-Block
import os
from oft.utils.config import load_config


class ObjectFusionTransformerDecoder(nn.Module):
    """
    Transformer-Decoder zur Vorhersage einer Menge von Objekten basierend auf
    Encoder-Memory und lernbaren Objekt-Queries.
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
                 box_dim: int = 7 # cx, cy, cz, w, l, h, yaw       
                ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.box_dim = box_dim # Speichern für den Forward-Pass

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

        self.class_head = nn.Linear(d_model, num_classes + 1) 
        
        # Bbox_head: MLP zur Vorhersage der Box-Parameter
        # Die letzte Schicht gibt `box_dim` Werte aus.
        # Wir werden Aktivierungsfunktionen im forward-Pass anwenden.
        self.bbox_head_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, box_dim) 
        )

    def forward(self,
                memory: torch.Tensor, 
                memory_key_padding_mask: Optional[torch.Tensor] = None
               ) -> Dict[str, torch.Tensor]:
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
        
        # Bounding-Box-Vorhersagen (rohe Ausgaben des MLP)
        raw_pred_boxes = self.bbox_head_mlp(decoder_output) # (B, NumQueries, box_dim)
        
        # Anwendung von Aktivierungsfunktionen auf die Box-Parameter
        # cx, cy, cz (Indizes 0, 1, 2) bleiben unbeschränkt (oder werden relativ zu Referenzpunkten gelernt, hier nicht der Fall)
        pred_cxcycz = raw_pred_boxes[..., :3]
        
        # w, l, h (Indizes 3, 4, 5) müssen positiv sein. torch.exp() ist eine gängige Wahl.
        # Vorsicht: exp kann explodieren. Softplus oder Sigmoid * Skalierungsfaktor wären Alternativen.
        # Für den Anfang versuchen wir es mit exp, da es in vielen Modellen verwendet wird.
        pred_wlh = torch.exp(raw_pred_boxes[..., 3:6]) 
        
        # yaw (Index 6) sollte im Bereich [-pi, pi] liegen.
        # torch.tanh gibt Werte in [-1, 1] aus. Multiplikation mit math.pi skaliert auf [-pi, pi].
        pred_yaw = torch.tanh(raw_pred_boxes[..., 6:7]) * math.pi # Stelle sicher, dass es (B, NumQueries, 1) bleibt
        
        # Kombiniere die verarbeiteten Box-Parameter
        pred_boxes = torch.cat((pred_cxcycz, pred_wlh, pred_yaw), dim=-1) # (B, NumQueries, 7)
        
        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes
        }

if __name__ == '__main__':
    print("Running ObjectFusionTransformerDecoder example with config-loaded parameters...")

    # --- Lade Konfiguration ---
    config_file_path = "config/pipeline_c_modules.yaml"
    # Fallback, falls die Haupt-Konfig nicht existiert (für isolierten Test)
    if not os.path.exists(config_file_path):
        alt_config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'pipeline_c_modules.yaml')
        if os.path.exists(alt_config_path):
            config_file_path = alt_config_path
        else:
            print(f"WARNUNG: Config file not found at {config_file_path} or {alt_config_path}. Using default test parameters.")
            config_file_path = None # Signalisiert, dass Fallback verwendet werden soll

    if config_file_path:
        try:
            full_pipeline_config = load_config(config_file_path)
            print(f"Konfiguration für Decoder-Test geladen von: {os.path.abspath(config_file_path)}")
            model_cfg = full_pipeline_config.get('model', {})
        except Exception as e:
            print(f"Fehler beim Laden der Konfiguration für Decoder-Test: {e}. Verwende Standard-Fallback-Parameter.")
            model_cfg = {} # Leeres Dict, um Defaults auszulösen
    else:
        model_cfg = {} # Leeres Dict, um Defaults auszulösen

    # Extrahiere Parameter für den Decoder oder verwende Defaults
    d_model_cfg = model_cfg.get('d_model', 256)
    n_heads_cfg = model_cfg.get('nhead', 8)
    num_dec_layers_cfg = model_cfg.get('num_decoder_layers', 3) # Weniger Layer für schnellen Test
    dim_ff_decoder_cfg = model_cfg.get('dim_feedforward_decoder', d_model_cfg * 4)
    dropout_cfg = model_cfg.get('dropout', 0.1)
    activation_cfg = model_cfg.get('activation', 'relu')
    num_queries_cfg = model_cfg.get('num_queries', 100)
    num_classes_cfg = model_cfg.get('num_classes', 5) # Kleinere Anzahl Klassen für Test
    box_dim_cfg = model_cfg.get('box_dim', 7)
    
    batch_s = 2
    num_enc_objects = 50 
    
    dummy_encoder_memory = torch.rand(batch_s, num_enc_objects, d_model_cfg)
    dummy_memory_padding_mask = torch.zeros(batch_s, num_enc_objects, dtype=torch.bool)
    if num_enc_objects > 10:
        dummy_memory_padding_mask[:, -10:] = True

    print(f"\nVerwendete Decoder-Parameter:")
    print(f"  d_model: {d_model_cfg}, nhead: {n_heads_cfg}, num_decoder_layers: {num_dec_layers_cfg}")
    print(f"  dim_feedforward: {dim_ff_decoder_cfg}, dropout: {dropout_cfg}, activation: {activation_cfg}")
    print(f"  num_queries: {num_queries_cfg}, num_classes (ohne BG): {num_classes_cfg}, box_dim: {box_dim_cfg}")

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
    print(f"  pred_boxes shape: {predictions['pred_boxes'].shape}")   

    assert predictions['pred_logits'].shape == (batch_s, num_queries_cfg, num_classes_cfg + 1)
    assert predictions['pred_boxes'].shape == (batch_s, num_queries_cfg, box_dim_cfg)
    
    print(f"\nBeispiel Output für erste Query, erstes Sample (nach Aktivierungen):")
    print(f"  Logits (erste 5 Werte): {predictions['pred_logits'][0, 0, :5].tolist()}")
    print(f"  Box (alle {box_dim_cfg} Werte): {np.round(predictions['pred_boxes'][0, 0, :].cpu().numpy(), 3)}")
    print(f"    cx,cy,cz: {np.round(predictions['pred_boxes'][0, 0, :3].cpu().numpy(), 3)}")
    print(f"    w,l,h (nach exp): {np.round(predictions['pred_boxes'][0, 0, 3:6].cpu().numpy(), 3)}")
    print(f"    yaw (nach tanh*pi): {predictions['pred_boxes'][0, 0, 6].item():.3f} rad")
    
    print("\nObjectFusionTransformerDecoder example run successful.")