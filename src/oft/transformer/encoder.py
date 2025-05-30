# src/oft/transformer/encoder.py
import torch
import torch.nn as nn
import math
from typing import Optional

class PositionalEncoding3D(nn.Module):
    """
    Implementiert eine feste sinusiodale 3D-Positionskodierung.
    Die Kodierung wird für jede der x, y, z Koordinaten separat berechnet
    und dann konkateniert, um die Zieldimension d_model zu erreichen.
    """
    def __init__(self, 
                 d_model: int, 
                 dropout: float = 0.1, 
                 max_coord_val: float = 150.0, # Max. erwarteter Absolutwert für Koord. für PE-Skalierung
                 num_freq_bands_per_coord: Optional[int] = None): 
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        if d_model % 6 != 0:
            # Jede der 3 Dimensionen (x,y,z) bekommt d_model/3 Features.
            # Jedes Feature-Paar (sin, cos) benötigt 2 Dimensionen.
            raise ValueError(f"d_model ({d_model}) muss durch 6 teilbar sein für 3D PE (x,y,z je sin/cos).")
        
        self.dim_per_coord = d_model // 3 # Anzahl der Features pro Koordinate (x, y, oder z)
        if self.dim_per_coord % 2 != 0:
            raise ValueError(f"d_model // 3 ({self.dim_per_coord}) muss gerade sein für Sin/Cos Paare.")

        if num_freq_bands_per_coord is None or num_freq_bands_per_coord <= 0:
            # Jedes Frequenzband erzeugt 2 Features (sin, cos)
            self.num_freq_bands_per_coord = self.dim_per_coord // 2
        else:
            self.num_freq_bands_per_coord = num_freq_bands_per_coord
        
        if self.num_freq_bands_per_coord * 2 != self.dim_per_coord:
            # Diese Bedingung stellt sicher, dass die Anzahl der Bänder genau die Hälfte der Features pro Koordinate erzeugt.
            raise ValueError(f"Die Anzahl der Frequenzbänder ({self.num_freq_bands_per_coord}) mal 2 (= {self.num_freq_bands_per_coord*2}) "
                             f"passt nicht zur Zieldimension pro Koordinate ({self.dim_per_coord}).")

        # Erzeuge Frequenzbänder: z.B. [1, 2, 4, 8, ...] skaliert mit pi / max_coord_val
        log_freq_bands = torch.arange(self.num_freq_bands_per_coord, dtype=torch.float32)
        # Frequenzen, die exponentiell ansteigen. Geteilt durch max_coord_val, um die Eingabekoordinaten
        # implizit zu normalisieren, bevor sie in sin/cos gehen.
        self.register_buffer('freq_bands', (math.pi / max_coord_val) * (2.0 ** log_freq_bands))


    def forward(self, xyz_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz_coords: Tensor der Form (Batch, NumObjects, 3) mit x,y,z Koordinaten.
                        Oder (NumObjects, 3) wenn Batch-Dimension fehlt.
        Returns:
            Tensor der Form (Batch, NumObjects, d_model) oder (NumObjects, d_model)
            mit Positions-Encodings.
        """
        original_ndim = xyz_coords.ndim
        if original_ndim == 2: # Einzelnes Sample ohne Batch-Dimension
            xyz_coords = xyz_coords.unsqueeze(0) 
        
        B, S, C = xyz_coords.shape
        if C != 3:
            raise ValueError("Input xyz_coords muss 3 Kanäle (x,y,z) haben.")

        # Erweitere xyz_coords und freq_bands für Broadcasting
        # xyz_coords: (B, S, 3) -> (B, S, 3, 1)
        # freq_bands: (num_freq_bands_per_coord) -> (1, 1, 1, num_freq_bands_per_coord)
        # scaled_inputs wird (B, S, 3, num_freq_bands_per_coord)
        scaled_inputs = xyz_coords.unsqueeze(-1) * self.freq_bands.view(1, 1, 1, -1)
        
        # Sinus und Cosinus anwenden und konkatenieren entlang der letzten Dimension
        # encoded_components wird (B, S, 3, num_freq_bands_per_coord * 2)
        encoded_components = torch.cat([torch.sin(scaled_inputs), torch.cos(scaled_inputs)], dim=-1)
        
        # Reshape, um die Encodings für x, y, z entlang der Feature-Dimension zu konkatenieren
        # Ziel: (B, S, 3 * (num_freq_bands_per_coord * 2)) = (B, S, d_model)
        output = encoded_components.view(B, S, self.d_model)
        
        if original_ndim == 2: # Entferne Batch-Dimension, falls sie ursprünglich nicht da war
            output = output.squeeze(0)
            
        return self.dropout(output)


class ObjectEncoder(nn.Module):
    """
    Einfacher Transformer-Encoder für die Verarbeitung von Objekt-Detektionslisten
    eines einzelnen Zeitpunkts.
    
    Implementiert:
    1. Basis-Feature-Aufbereitung ("Environmental Attention" im Sinne der Objektmerkmale):
       - Input-Projektion der rohen Detektionsfeatures (Box, Geschwindigkeit) auf d_model.
       - Hinzufügen von 3D Positional Encoding basierend auf den Objektzentren.
    2. Kontextualisierung durch Self-Attention ("Intra-Modal Self-Attention"):
       - Standard Transformer Encoder Schichten, die Beziehungen zwischen den Objekten
         innerhalb des aktuellen Frames lernen.
    
    Dieser Encoder dient als Grundlage und kann später für komplexere Interaktionen
    (Inter-Modal, Temporal) erweitert werden.
    """
    def __init__(self,
                 num_input_features: int = 10, # z.B. 7 für Box + 3 für Velocity
                 d_model: int = 256,
                 nhead: int = 8, # Anzahl der Attention Heads
                 num_encoder_layers: int = 2, # Anzahl der Self-Attention Schichten
                 dim_feedforward: int = 1024, # Dimension des Feed-Forward Netzwerks in jeder Schicht
                 dropout: float = 0.1,
                 activation: str = "relu", # Aktivierungsfunktion in den Feed-Forward Netzwerken
                 pe_max_coord_val: float = 150.0, # Für PositionalEncoding3D
                 pe_num_freq_bands: Optional[int] = None # Für PositionalEncoding3D
                ):
        super().__init__()
        self.d_model = d_model

        # Stufe 1a: Input-Projektion der rohen Objektmerkmale
        self.input_projection = nn.Linear(num_input_features, d_model)
        
        # Stufe 1b: 3D Positional Encoding für die Objektzentren
        self.positional_encoding = PositionalEncoding3D(
            d_model=d_model, 
            dropout=dropout,
            max_coord_val=pe_max_coord_val,
            num_freq_bands_per_coord=pe_num_freq_bands
        )
        
        # Normalisierung und Dropout nach Kombination von Features und PE
        self.norm_after_pe_add = nn.LayerNorm(d_model) 
        self.dropout_after_pe_add = nn.Dropout(dropout)

        # Stufe 2: Intra-Modal Self-Attention Schichten
        # Jede Schicht besteht aus Multi-Head Self-Attention und einem Feed-Forward Netzwerk
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True # Eingabeformat ist (Batch, Sequenz, Feature)
        )
        self.transformer_encoder_stack = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model) # Optionale finale LayerNorm nach allen Schichten
        )
        

    def forward(self, 
                src_features: torch.Tensor, 
                src_xyz_centers: torch.Tensor, # KORRIGIERTE REIHENFOLGE
                src_padding_mask: Optional[torch.Tensor] = None # Argument mit Defaultwert jetzt am Ende
               ) -> torch.Tensor:
        """
        Forward-Pass des Encoders.
        
        Args:
            src_features (torch.Tensor): Rohe Features der Detektionen. 
                                         Shape: (Batch, NumObjects, num_input_features)
            src_xyz_centers (torch.Tensor): XYZ-Zentren der Detektionen für Positional Encoding.
                                            Shape: (Batch, NumObjects, 3)
            src_padding_mask (Optional[torch.Tensor]): Maske für Padding-Token.
                                                       Shape: (Batch, NumObjects), True für Padding.
            
        Returns:
            memory (torch.Tensor): Encoder-Output, kontextualisierte Repräsentationen der Objekte.
                                   Shape: (Batch, NumObjects, d_model)
        """
        
        # 1. Input-Projektion (Basis-Feature-Aufbereitung, Teil 1)
        projected_src = self.input_projection(src_features) # (B, S, d_model)
        
        # 2. 3D Positional Encoding (Basis-Feature-Aufbereitung, Teil 2)
        pos_enc = self.positional_encoding(src_xyz_centers) # (B, S, d_model)
        
        # Kombiniere projizierte Features mit Positions-Encoding
        x = projected_src + pos_enc # Additive Kombination
        x = self.norm_after_pe_add(x) 
        x = self.dropout_after_pe_add(x)

        # 3. Transformer Encoder Stack (Intra-Modal Self-Attention)
        # Verarbeitet die Sequenz von aufbereiteten Objekt-Tokens.
        # src_key_padding_mask wird verwendet, um Padding-Positionen in der Attention zu ignorieren.
        memory = self.transformer_encoder_stack(
            src=x,
            src_key_padding_mask=src_padding_mask 
        )
        # Output: memory (B, S, d_model)
        return memory

if __name__ == '__main__':
    print("Running ObjectEncoder (minimal version for single timestamp) example...")

    # Beispiel-Hyperparameter
    batch_size = 2
    max_objects_per_sample = 8 
    num_input_features = 10    
    d_model = 128              
    n_heads = 4                
    num_enc_layers = 2         
    dim_ff = d_model * 4       

    dummy_src_features = torch.rand(batch_size, max_objects_per_sample, num_input_features)
    dummy_src_xyz_centers = torch.rand(batch_size, max_objects_per_sample, 3) * 100.0 - 50.0 

    dummy_src_padding_mask = torch.ones(batch_size, max_objects_per_sample, dtype=torch.bool)
    if max_objects_per_sample >= 5: dummy_src_padding_mask[0, :5] = False
    if max_objects_per_sample >= 3: dummy_src_padding_mask[1, :3] = False
    
    print(f"\nDummy Input Shapes:")
    print(f"  src_features: {dummy_src_features.shape}")
    print(f"  src_xyz_centers: {dummy_src_xyz_centers.shape}")
    print(f"  src_padding_mask: {dummy_src_padding_mask.shape}")

    encoder = ObjectEncoder(
        num_input_features=num_input_features,
        d_model=d_model,
        nhead=n_heads,
        num_encoder_layers=num_enc_layers,
        dim_feedforward=dim_ff,
        pe_max_coord_val=150.0, 
        pe_num_freq_bands=None  
    )
    encoder.eval() 

    with torch.no_grad(): 
        output_memory = encoder(
            src_features=dummy_src_features,
            src_xyz_centers=dummy_src_xyz_centers, # Angepasste Reihenfolge
            src_padding_mask=dummy_src_padding_mask
        )

    print(f"\nEncoder Output Shape (memory): {output_memory.shape}")
    assert output_memory.shape == (batch_size, max_objects_per_sample, d_model)
    
    print(f"Output für Sample 0, erstes echtes Objekt (Ausschnitt): {output_memory[0, 0, :5].tolist()}")
    if max_objects_per_sample > 5 :
        print(f"Output für Sample 0, erstes gepaddetes Objekt (Ausschnitt): {output_memory[0, 5, :5].tolist()}")
    
    print("\nObjectEncoder example run successful.")