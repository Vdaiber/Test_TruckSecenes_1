# src/oft/transformer/decoder.py
import torch
import torch.nn as nn
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
                 box_dim: int = 7        
                ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

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
        self.bbox_head = nn.Sequential(
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
        tgt = self.object_queries_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        decoder_output = self.transformer_decoder_stack(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=None, 
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        pred_logits = self.class_head(decoder_output)    
        pred_boxes = self.bbox_head(decoder_output)      
        
        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes
        }

if __name__ == '__main__':
    print("Running ObjectFusionTransformerDecoder example with config-loaded parameters...")

    # --- ANPASSUNG START: Lade Konfiguration ---
    config_file_path = "config/pipeline_c_modules.yaml"
    try:
        full_pipeline_config = load_config(config_file_path)
        print(f"Konfiguration für Decoder-Test geladen von: {os.path.abspath(config_file_path)}")
        
        model_cfg = full_pipeline_config.get('model', {})
        
        # Extrahiere Parameter für den Decoder
        d_model_cfg = model_cfg.get('d_model', 256)
        n_heads_cfg = model_cfg.get('nhead', 8)
        num_dec_layers_cfg = model_cfg.get('num_decoder_layers', 6)
        dim_ff_decoder_cfg = model_cfg.get('dim_feedforward_decoder', d_model_cfg * 4)
        dropout_cfg = model_cfg.get('dropout', 0.1)
        activation_cfg = model_cfg.get('activation', 'relu')
        num_queries_cfg = model_cfg.get('num_queries', 100)
        num_classes_cfg = model_cfg.get('num_classes', 10) # Ohne Hintergrundklasse
        box_dim_cfg = model_cfg.get('box_dim', 7)

    except Exception as e:
        print(f"Fehler beim Laden der Konfiguration für Decoder-Test: {e}")
        print("Verwende Standard-Fallback-Parameter.")
        d_model_cfg = 256                
        n_heads_cfg = 8               
        num_dec_layers_cfg = 3 # Weniger Layer für schnellen Fallback         
        dim_ff_decoder_cfg = d_model_cfg * 4       
        dropout_cfg = 0.1
        activation_cfg = "relu"
        num_queries_cfg = 100          
        num_classes_cfg = 5 # Kleinere Anzahl Klassen für Fallback         
        box_dim_cfg = 7 
    # --- ANPASSUNG ENDE ---

    batch_s = 2
    num_enc_objects = 50 # Anzahl der Objekte im Encoder-Output (Memory)
    
    # Dummy Encoder-Output (Memory)
    # Die Dimension d_model_cfg muss mit dem Encoder-Output übereinstimmen
    dummy_encoder_memory = torch.rand(batch_s, num_enc_objects, d_model_cfg)
    # Dummy Padding-Maske für die Encoder-Memory
    dummy_memory_padding_mask = torch.zeros(batch_s, num_enc_objects, dtype=torch.bool)
    if num_enc_objects > 10:
        dummy_memory_padding_mask[:, -10:] = True # Die letzten 10 Objekte im Encoder-Output sind gepadded

    print(f"\nVerwendete Decoder-Parameter (aus Config oder Fallback):")
    print(f"  d_model: {d_model_cfg}")
    print(f"  nhead: {n_heads_cfg}")
    print(f"  num_decoder_layers: {num_dec_layers_cfg}")
    print(f"  dim_feedforward: {dim_ff_decoder_cfg}")
    print(f"  dropout: {dropout_cfg}")
    print(f"  activation: {activation_cfg}")
    print(f"  num_queries: {num_queries_cfg}")
    print(f"  num_classes (ohne BG): {num_classes_cfg}")
    print(f"  box_dim: {box_dim_cfg}")

    print(f"\nDummy Input Shapes für Decoder:")
    print(f"  Encoder Memory (memory): {dummy_encoder_memory.shape}")
    print(f"  Memory Padding Mask (memory_key_padding_mask): {dummy_memory_padding_mask.shape}")
    
    decoder = ObjectFusionTransformerDecoder(
        d_model=d_model_cfg,
        nhead=n_heads_cfg,
        num_decoder_layers=num_dec_layers_cfg, 
        dim_feedforward=dim_ff_decoder_cfg,
        dropout=dropout_cfg,
        activation=activation_cfg,
        num_queries=num_queries_cfg,
        num_classes=num_classes_cfg,
        box_dim=box_dim_cfg
    )
    decoder.eval()

    with torch.no_grad():
        predictions = decoder(
            memory=dummy_encoder_memory,
            memory_key_padding_mask=dummy_memory_padding_mask
        )

    print(f"\nDecoder Output Dictionary Keys: {list(predictions.keys())}")
    print(f"  pred_logits shape: {predictions['pred_logits'].shape}") # Sollte (B, num_queries, num_classes + 1) sein
    print(f"  pred_boxes shape: {predictions['pred_boxes'].shape}")   # Sollte (B, num_queries, box_dim) sein

    assert predictions['pred_logits'].shape == (batch_s, num_queries_cfg, num_classes_cfg + 1)
    assert predictions['pred_boxes'].shape == (batch_s, num_queries_cfg, box_dim_cfg)
    
    print(f"\nOutput für erste Query, erstes Sample (Ausschnitt):")
    print(f"  Logits (erste 5 Werte): {predictions['pred_logits'][0, 0, :5].tolist()}")
    print(f"  Box (alle {box_dim_cfg} Werte): {predictions['pred_boxes'][0, 0, :].tolist()}")
    
    print("\nObjectFusionTransformerDecoder example run successful.")