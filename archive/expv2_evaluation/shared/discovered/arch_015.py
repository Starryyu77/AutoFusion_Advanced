import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=418, num_layers=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Projection layers
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        
        # Multi-layer attention blocks with gating
        self.attention_layers = nn.ModuleList()
        self.gate_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            # Cross-attention layer
            self.attention_layers.append(
                nn.MultiheadAttention(hidden_dim, num_heads=2, batch_first=True)
            )
            
            # Gating mechanism
            self.gate_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Sigmoid()
                )
            )
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, vision_features, language_features):
        # Project features to hidden dimension
        v = self.vision_proj(vision_features)
        l = self.language_proj(language_features)
        
        # Process through multiple attention-gating layers
        for attn_layer, gate_layer in zip(self.attention_layers, self.gate_layers):
            # Cross-attention: language attends to vision
            attn_out, _ = attn_layer(
                l.unsqueeze(1), 
                v.unsqueeze(1), 
                v.unsqueeze(1)
            )
            attn_out = attn_out.squeeze(1)
            
            # Gated fusion
            gate = gate_layer(torch.cat([attn_out, l], dim=-1))
            l = gate * attn_out + (1 - gate) * l
        
        # Normalize and return
        return self.norm(l)