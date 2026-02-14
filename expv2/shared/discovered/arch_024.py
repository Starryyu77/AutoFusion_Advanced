import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=218, num_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Projection layers
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        
        # Bilinear fusion layer
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        
        # Transformer layers for hybrid processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=2,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, vision_features, language_features):
        # Project features to hidden dimension
        v = self.vision_proj(vision_features)
        l = self.language_proj(language_features)
        
        # Bilinear fusion
        fused = self.bilinear(v, l)
        
        # Add residual connection
        fused = fused + v + l
        
        # Apply transformer layers (hybrid processing)
        fused = fused.unsqueeze(1)
        fused = self.transformer(fused)
        fused = fused.squeeze(1)
        
        # Normalize and project
        fused = self.norm(fused)
        fused = self.output_proj(fused)
        
        return fused