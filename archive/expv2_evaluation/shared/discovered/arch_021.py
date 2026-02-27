import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=1024, num_layers=4):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        
        layers = []
        for i in range(num_layers):
            in_dim = hidden_dim * 2 if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        
        self.fusion_mlp = nn.Sequential(*layers)

    def forward(self, vision_features, language_features):
        v = self.vision_proj(vision_features)
        l = self.language_proj(language_features)
        fused = torch.cat([v, l], dim=-1)
        return self.fusion_mlp(fused)