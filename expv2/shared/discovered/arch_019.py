import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=852, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Project both modalities to hidden dimension
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=12, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # MLP fusion layers
        self.mlp_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim * 2 if i == 0 else hidden_dim
            out_dim = hidden_dim
            self.mlp_layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
            )
        
        # Final projection
        self.final_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_features, language_features):
        # Project features
        v = self.vision_proj(vision_features).unsqueeze(1)  # [B, 1, D]
        l = self.language_proj(language_features).unsqueeze(1)  # [B, 1, D]
        
        # Cross-modal attention
        attn_out, _ = self.attention(l, v, v)  # Language attends to vision
        attn_out = self.attention_norm(attn_out.squeeze(1) + l.squeeze(1))
        
        # MLP fusion
        fused = torch.cat([v.squeeze(1), attn_out], dim=-1)
        for mlp_layer in self.mlp_layers:
            fused = mlp_layer(fused)
        
        # Final processing
        fused = self.final_proj(fused)
        fused = self.norm(fused)
        
        return fused