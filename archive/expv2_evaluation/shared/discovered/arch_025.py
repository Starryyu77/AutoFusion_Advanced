import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=211, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Projection layers
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        
        # Multi-head attention for cross-modal fusion
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=1,  # 211 is prime, using 1 head for simplicity
            batch_first=True,
            dropout=0.1
        )
        
        # Transformer layers for deeper fusion
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=1,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Residual projection if dimensions don't match
        self.residual_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, vision_features, language_features):
        # Project to common hidden dimension
        v = self.vision_proj(vision_features).unsqueeze(1)  # (batch, 1, hidden_dim)
        l = self.language_proj(language_features).unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Cross-attention: language attends to vision
        attn_out, _ = self.cross_attn(l, v, v)
        attn_out = self.norm1(attn_out + l)  # Residual connection
        
        # Concatenate for transformer processing
        concat = torch.cat([v, attn_out], dim=1)  # (batch, 2, hidden_dim)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            concat = layer(concat)
        
        # Average pooling across sequence dimension
        fused = concat.mean(dim=1)  # (batch, hidden_dim)
        
        # Gated fusion of original projections
        v_flat = v.squeeze(1)
        l_flat = l.squeeze(1)
        gate = self.gate(torch.cat([v_flat, l_flat], dim=-1))
        gated_fusion = gate * v_flat + (1 - gate) * l_flat
        
        # Final fusion with residual connection
        final_fused = self.norm2(fused + gated_fusion)
        
        return final_fused