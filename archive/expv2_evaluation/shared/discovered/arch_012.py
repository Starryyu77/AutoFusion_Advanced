import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=458, num_layers=3):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=num_layers
        )
        
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_features, language_features):
        v = self.vision_proj(vision_features).unsqueeze(1)
        l = self.language_proj(language_features).unsqueeze(1)
        
        concat = torch.cat([v, l], dim=1)
        transformer_out = self.transformer(concat)
        
        fused = transformer_out.mean(dim=1)
        return self.norm(fused)