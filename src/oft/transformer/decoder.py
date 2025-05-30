# src/oft/transformer/decoder.py
import torch
import torch.nn as nn
from typing import Optional, Dict, List # Dict und List hinzugefügt

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
                 num_queries: int = 100, # Maximale Anzahl vorhergesagter Objekte
                 num_classes: int = 10,  # Anzahl der Objektklassen (ohne "kein Objekt")
                 box_dim: int = 7        # Dimension der Box-Vorhersage (z.B. x,y,z,w,l,h,yaw)
                ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        # Lernbare Objekt-Queries
        # Diese Queries sind lernbare Embeddings, die als "Slots" für vorhergesagte Objekte dienen.
        # Shape: (num_queries, d_model)
        self.object_queries_embed = nn.Embedding(num_queries, d_model)
        # Alternativ: nn.Parameter(torch.randn(num_queries, d_model))

        # Standard Transformer Decoder Schicht
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True # Eingabeformat ist (Batch, Sequenz, Feature)
        )
        # Stapel von Decoder Schichten
        self.transformer_decoder_stack = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model) # Finale LayerNorm nach allen Decoder-Schichten
        )

        # Prädiktionsköpfe
        # Klassifikationskopf: Sagt die Klasse für jede Query voraus (+1 für "kein Objekt" Klasse)
        self.class_head = nn.Linear(d_model, num_classes + 1) 
        
        # Box-Regressionskopf: Sagt die Box-Parameter für jede Query voraus
        # Ein kleines MLP ist hier oft besser als ein einzelner linearer Layer
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), # Reduziere Dimension
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, box_dim) 
            # Keine Aktivierung am Ende für Box-Regression, außer Sigmoid für normalisierte cxcywh
            # Für Weltkoordinaten ist keine End-Aktivierung üblich.
        )

    def forward(self,
                memory: torch.Tensor, # Output vom Encoder (C2), Shape: (B, S_encoder, d_model)
                memory_key_padding_mask: Optional[torch.Tensor] = None # Maske für Encoder-Output, Shape: (B, S_encoder)
               ) -> Dict[str, torch.Tensor]: # Hier wurde Dict verwendet
        """
        Forward-Pass des Decoders.

        Args:
            memory (torch.Tensor): Die kontextualisierten Repräsentationen vom Encoder.
            memory_key_padding_mask (Optional[torch.Tensor]): Padding-Maske für die Encoder-Memory.
                                                              (True an Stellen, die ignoriert werden sollen).
        Returns:
            Dict[str, torch.Tensor]: Ein Dictionary mit:
                - "pred_logits": Logits für die Klassifikation jeder Query. 
                                 Shape: (B, num_queries, num_classes + 1)
                - "pred_boxes": Vorhergesagte Box-Parameter für jede Query.
                                Shape: (B, num_queries, box_dim)
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
        pred_boxes = self.bbox_head(decoder_output)      
        
        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes
        }

if __name__ == '__main__':
    print("Running ObjectFusionTransformerDecoder example...")

    batch_s = 2
    num_enc_objects = 50 
    d_model_size = 256
    num_q = 100          
    num_cls = 5          
    box_prediction_dim = 7 

    dummy_encoder_memory = torch.rand(batch_s, num_enc_objects, d_model_size)
    dummy_memory_padding_mask = torch.zeros(batch_s, num_enc_objects, dtype=torch.bool)
    if num_enc_objects > 10:
        dummy_memory_padding_mask[:, -10:] = True 

    print(f"\nDummy Input Shapes:")
    print(f"  Encoder Memory: {dummy_encoder_memory.shape}")
    print(f"  Memory Padding Mask (True=Padding): {dummy_memory_padding_mask.shape}")
    
    decoder = ObjectFusionTransformerDecoder(
        d_model=d_model_size,
        nhead=8,
        num_decoder_layers=3, 
        dim_feedforward=d_model_size * 4,
        num_queries=num_q,
        num_classes=num_cls,
        box_dim=box_prediction_dim
    )
    decoder.eval()

    with torch.no_grad():
        predictions = decoder(
            memory=dummy_encoder_memory,
            memory_key_padding_mask=dummy_memory_padding_mask
        )

    print(f"\nDecoder Output Dictionary Keys: {predictions.keys()}")
    print(f"  pred_logits shape: {predictions['pred_logits'].shape}")
    print(f"  pred_boxes shape: {predictions['pred_boxes'].shape}")

    assert predictions['pred_logits'].shape == (batch_s, num_q, num_cls + 1)
    assert predictions['pred_boxes'].shape == (batch_s, num_q, box_prediction_dim)
    
    print(f"\nOutput für erste Query, erstes Sample (Ausschnitt):")
    print(f"  Logits: {predictions['pred_logits'][0, 0, :].tolist()}")
    print(f"  Box: {predictions['pred_boxes'][0, 0, :].tolist()}")
    
    print("\nObjectFusionTransformerDecoder example run successful.")