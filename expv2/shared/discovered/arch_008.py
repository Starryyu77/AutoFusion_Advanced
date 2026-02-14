import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=467, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Projection layers
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        
        # Bilinear fusion layer
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        
        # Multi-layer processing
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
            )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_features, language_features):
        # Project features to hidden dimension
        v = self.vision_proj(vision_features)
        l = self.language_proj(language_features)
        
        # Bilinear fusion
        fused = self.bilinear(v, l)
        
        # Process through multiple layers
        for layer in self.layers:
            fused = layer(fused)
        
        # Final projection and normalization
        output = self.output_proj(fused)
        output = self.layer_norm(output)
        
        return output