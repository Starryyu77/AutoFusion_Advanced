import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=831, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Project both modalities to hidden dimension
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=3, batch_first=True)
        
        # MLP fusion layers
        self.mlp_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim * 2 if i == 0 else hidden_dim
            out_dim = hidden_dim
            self.mlp_layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1)
                )
            )
        
        # Final projection
        self.final_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_features, language_features):
        # Project to common dimension
        v = self.vision_proj(vision_features)
        l = self.language_proj(language_features)
        
        # Apply cross-attention (language attends to vision)
        v_expanded = v.unsqueeze(1)
        l_expanded = l.unsqueeze(1)
        attn_out, _ = self.attention(l_expanded, v_expanded, v_expanded)
        attn_out = attn_out.squeeze(1)
        
        # MLP-based fusion
        fused = torch.cat([attn_out, l], dim=-1)
        for mlp_layer in self.mlp_layers:
            fused = mlp_layer(fused)
        
        # Final processing
        fused = self.final_proj(fused)
        fused = self.norm(fused)
        
        return fused