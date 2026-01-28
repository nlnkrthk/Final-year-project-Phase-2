import torch
import torch.nn as nn

class BaselineClassifier(nn.Module):
    def __init__(self, input_dim=1536):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
