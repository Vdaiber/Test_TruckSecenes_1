# src/oft/transformer/encoder.py
import torch
import torch.nn as nn
import math
from typing import Optional

# NEU: Importe für das Laden der Konfiguration im Test-Block
import os
# from oft.utils.config import load_config # Wird nur im if __name__ Block benötigt


class PositionalEncoding3D(nn.Module):
    """
    Implementiert eine feste sinusiodale 3D-Positionskodierung.
    Die Kodierung wird für jede der x, y, z Koordinaten separat berechnet
    und dann konkateniert, um die Zieldimension d_model zu erreichen.
    """
    def __init__(self, 
                 d_model: int, 
                 dropout: float = 0.1, 
                 max_coord_val: float = 150.0, 
                 num_freq_bands_per_coord: Optional[int] = None): 
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        if d_model % 6 != 0:
            raise ValueError(f"d_model ({d_model}) muss durch 6 teilbar sein für 3D PE (x,y,z je sin/cos).")
        
        self.dim_per_coord = d_model // 3 
        if self.dim_per_coord % 2 != 0:
            raise ValueError(f"d_model // 3 ({self.dim_per_coord}) muss gerade sein für Sin/Cos Paare.")

        if num_freq_bands_per_coord is None or num_freq_bands_per_coord <= 0:
            self.num_freq_bands_per_coord = self.dim_per_coord // 2
        else:
            self.num_freq_bands_per_coord = num_freq_bands_per_coord
        
        if self.num_freq_bands_per_coord * 2 != self.dim_per_coord:
            raise ValueError(f"Die Anzahl der Frequenzbänder ({self.num_freq_bands_per_coord}) mal 2 (= {self.num_freq_bands_per_coord*2}) "
                             f"passt nicht zur Zieldimension pro Koordinate ({self.dim_per_coord}).")

        log_freq_bands = torch.arange(self.num_freq_bands_per_coord, dtype=torch.float32)
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
        if original_ndim == 2: 
            xyz_coords = xyz_coords.unsqueeze(0) 
        
        B, S, C = xyz_coords.shape
        if C != 3:
            raise ValueError("Input xyz_coords muss 3 Kanäle (x,y,z) haben.")

        scaled_inputs = xyz_coords.unsqueeze(-1) * self.freq_bands.view(1, 1, 1, -1)
        encoded_components = torch.cat([torch.sin(scaled_inputs), torch.cos(scaled_inputs)], dim=-1)
        output = encoded_components.view(B, S, self.d_model)
        
        if original_ndim == 2: 
            output = output.squeeze(0)
            
        return self.dropout(output)


class ObjectEncoder(nn.Module):
    """
    Einfacher Transformer-Encoder für die Verarbeitung von Objekt-Detektionslisten
    eines einzelnen Zeitpunkts.
    """
    def __init__(self,
                 num_input_features: int = 10, 
                 d_model: int = 256,
                 nhead: int = 8, 
                 num_encoder_layers: int = 2, 
                 dim_feedforward: int = 1024, 
                 dropout: float = 0.1,
                 activation: str = "relu", 
                 pe_max_coord_val: float = 150.0, 
                 # pe_num_freq_bands wurde entfernt, da PositionalEncoding3D es intern berechnet
                ):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(num_input_features, d_model)
        
        self.positional_encoding = PositionalEncoding3D(
            d_model=d_model, 
            dropout=dropout,
            max_coord_val=pe_max_coord_val,
            # num_freq_bands_per_coord wird intern von PositionalEncoding3D basierend auf d_model berechnet
        )
        
        self.norm_after_pe_add = nn.LayerNorm(d_model) 
        self.dropout_after_pe_add = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True 
        )
        self.transformer_encoder_stack = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model) 
        )
        

    def forward(self, 
                src_features: torch.Tensor, 
                src_xyz_centers: torch.Tensor,
                src_padding_mask: Optional[torch.Tensor] = None
               ) -> torch.Tensor:
        projected_src = self.input_projection(src_features)
        pos_enc = self.positional_encoding(src_xyz_centers)
        
        x = projected_src + pos_enc
        x = self.norm_after_pe_add(x) 
        x = self.dropout_after_pe_add(x)

        memory = self.transformer_encoder_stack(
            src=x,
            src_key_padding_mask=src_padding_mask 
        )
        return memory

if __name__ == '__main__':
    print("Running ObjectEncoder example with config-loaded parameters...")
    from oft.utils.config import load_config # Import hier für den Testblock

    # --- ANPASSUNG START: Lade Konfiguration ---
    config_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'pipeline_c_modules.yaml')
    try:
        full_pipeline_config = load_config(config_file_path)
        print(f"Konfiguration für Encoder-Test geladen von: {os.path.abspath(config_file_path)}")
        
        model_cfg = full_pipeline_config.get('model', {})
        
        # Extrahiere Parameter für den Encoder und PE
        num_input_features_cfg = model_cfg.get('num_input_features', 10) # Default, falls nicht in Config
        d_model_cfg = model_cfg.get('d_model', 256)
        n_heads_cfg = model_cfg.get('nhead', 8)
        num_enc_layers_cfg = model_cfg.get('num_encoder_layers', 2)
        dim_ff_cfg = model_cfg.get('dim_feedforward_encoder', d_model_cfg * 4) # Berechne, falls nicht explizit
        dropout_cfg = model_cfg.get('dropout', 0.1)
        activation_cfg = model_cfg.get('activation', 'relu')
        pe_max_coord_val_cfg = model_cfg.get('pe_max_coord_val', 150.0)
        # pe_num_freq_bands ist nicht mehr direkter Parameter für ObjectEncoder

    except Exception as e:
        print(f"Fehler beim Laden der Konfiguration für Encoder-Test: {e}")
        print("Verwende Standard-Fallback-Parameter.")
        num_input_features_cfg = 10    
        d_model_cfg = 128 # Kleinere Werte für schnellen Fallback-Test             
        n_heads_cfg = 4                
        num_enc_layers_cfg = 2         
        dim_ff_cfg = d_model_cfg * 4
        dropout_cfg = 0.1
        activation_cfg = "relu"
        pe_max_coord_val_cfg = 150.0
    # --- ANPASSUNG ENDE ---

    batch_size = 2
    max_objects_per_sample = 8 
    
    # Dummy-Daten basierend auf num_input_features_cfg
    dummy_src_features = torch.rand(batch_size, max_objects_per_sample, num_input_features_cfg)
    dummy_src_xyz_centers = torch.rand(batch_size, max_objects_per_sample, 3) * 100.0 - 50.0 

    dummy_src_padding_mask = torch.ones(batch_size, max_objects_per_sample, dtype=torch.bool)
    if max_objects_per_sample >= 5: dummy_src_padding_mask[0, :5] = False # Sample 0 hat 5 echte Objekte
    if max_objects_per_sample >= 3: dummy_src_padding_mask[1, :3] = False # Sample 1 hat 3 echte Objekte
    
    print(f"\nVerwendete Encoder-Parameter (aus Config oder Fallback):")
    print(f"  num_input_features: {num_input_features_cfg}")
    print(f"  d_model: {d_model_cfg}")
    print(f"  nhead: {n_heads_cfg}")
    print(f"  num_encoder_layers: {num_enc_layers_cfg}")
    print(f"  dim_feedforward: {dim_ff_cfg}")
    print(f"  dropout: {dropout_cfg}")
    print(f"  activation: {activation_cfg}")
    print(f"  pe_max_coord_val: {pe_max_coord_val_cfg}")

    print(f"\nDummy Input Shapes:")
    print(f"  src_features: {dummy_src_features.shape}")
    print(f"  src_xyz_centers: {dummy_src_xyz_centers.shape}")
    print(f"  src_padding_mask (True=Padding): {dummy_src_padding_mask.shape}, Echte Objekte: {[ (~mask).sum().item() for mask in dummy_src_padding_mask]}")

    encoder = ObjectEncoder(
        num_input_features=num_input_features_cfg,
        d_model=d_model_cfg,
        nhead=n_heads_cfg,
        num_encoder_layers=num_enc_layers_cfg,
        dim_feedforward=dim_ff_cfg,
        dropout=dropout_cfg,
        activation=activation_cfg,
        pe_max_coord_val=pe_max_coord_val_cfg
    )
    encoder.eval() 

    with torch.no_grad(): 
        output_memory = encoder(
            src_features=dummy_src_features,
            src_xyz_centers=dummy_src_xyz_centers,
            src_padding_mask=dummy_src_padding_mask
        )

    print(f"\nEncoder Output Shape (memory): {output_memory.shape}")
    assert output_memory.shape == (batch_size, max_objects_per_sample, d_model_cfg)
    
    print(f"Output für Sample 0, erstes echtes Objekt (Ausschnitt): {output_memory[0, 0, :5].tolist()}")
    if max_objects_per_sample > 5 : # Nur wenn es tatsächlich Padding gibt im ersten Sample
        if not dummy_src_padding_mask[0,5]: # Wenn Objekt 5 existiert (nicht gepadded ist)
             print(f"Output für Sample 0, sechstes Objekt (Index 5) (Ausschnitt): {output_memory[0, 5, :5].tolist()}")
        else: # Wenn Objekt 5 gepadded ist
             print(f"Output für Sample 0, erstes gepaddetes Objekt (Index 5) (Ausschnitt): {output_memory[0, 5, :5].tolist()}")
    
    print("\nObjectEncoder example run successful.")