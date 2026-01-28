import torch
import torch.nn as nn

class AttentionFusionModel(nn.Module):
    def __init__(self, dim=768, heads=4):
        super().__init__()

        self.audio_proj = nn.Linear(dim, dim)
        self.text_proj = nn.Linear(dim, dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, audio_feats, text_feats):
        """
        audio_feats: [B, T, 768]
        text_feats:  [B, C, 768]
        """

        audio = self.audio_proj(audio_feats)
        text = self.text_proj(text_feats)

        attended_audio, _ = self.cross_attention(
            query=audio,
            key=text,
            value=text
        )

        pooled = attended_audio.mean(dim=1)  # [B, 768]
        logits = self.classifier(pooled)

        return logits.squeeze(1)
