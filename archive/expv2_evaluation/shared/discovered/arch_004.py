import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=922, num_layers=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Project vision and language features to hidden dimension
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        
        # Attention mechanism for fusion
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # MLP layers for processing fused features
        mlp_layers = []
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(0.1))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_features, language_features):
        # Project features to hidden dimension
        v = self.vision_proj(vision_features)
        l = self.language_proj(language_features)
        
        # Compute attention weights for fusion
        attention_input = torch.cat([v, l], dim=-1)
        attention_weights = self.attention(attention_input)
        
        # Apply attention-based fusion
        v_weight = attention_weights[:, 0].unsqueeze(-1)
        l_weight = attention_weights[:, 1].unsqueeze(-1)
        fused = v_weight * v + l_weight * l
        
        # Process through MLP
        fused = self.mlp(fused)
        
        # Apply layer normalization
        fused = self.norm(fused)
        
        return fused