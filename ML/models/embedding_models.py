import torch.nn as nn

class EmbeddingModel(nn.Module):
    def __init__(self, input_dim=3, embedding_dim=16): # Small d for "psychological" space
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
    
    def forward(self, x):
        z = self.net(x)
        # L2 Normalization (Crucial for Contrastive/Triplet Loss)
        z = nn.functional.normalize(z, p=2, dim=1)
        return z
