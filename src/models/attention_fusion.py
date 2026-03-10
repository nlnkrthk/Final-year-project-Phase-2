import torch
import torch.nn as nn

class AttentionFusionModel(nn.Module):
    """
    Baseline classifier for Alzheimer's detection.
    No attention mechanism - focuses on robust feature fusion.
    Optimized for 4GB GPU memory.
    
    Input: 
    - audio_feats: [B, T, 768]
    - text_feats: [B, C, 768]
    
    Output:
    - logits: [B]
    """
    def __init__(self, dim=768, **kwargs):
        super().__init__()
        input_dim = dim * 2  # audio (768) + text (768)
        
        # Classifier head on concatenated features
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, audio_feats, text_feats):
        """
        Args:
            audio_feats: [B, T, 768] - audio frame features
            text_feats:  [B, C, 768] - text token features
        
        Returns:
            logits: [B] - classification logits
        """
        # Pool audio and text features
        audio_mean = audio_feats.mean(dim=1)  # [B, 768]
        audio_max = audio_feats.max(dim=1)[0]  # [B, 768]
        audio_pooled = audio_mean + audio_max  # [B, 768]
        
        text_mean = text_feats.mean(dim=1)    # [B, 768]
        text_max = text_feats.max(dim=1)[0]   # [B, 768]
        text_pooled = text_mean + text_max    # [B, 768]
        
        # Concatenate modalities
        fused = torch.cat([audio_pooled, text_pooled], dim=-1)  # [B, 1536]
        
        # Classification
        logits = self.classifier(fused)  # [B, 1]
        
        return logits.squeeze(1)
