import torch.nn as nn


class BasicChoiceModel(nn.Module):
    def __init__(self, embedding_dim=16, num_classes=10):
        super().__init__()
        # A simple linear probe or shallow MLP acts as the "Decision Boundary"
        self.net = nn.Linear(embedding_dim, num_classes)

    def forward(self, z):
        return self.net(z)  # Returns logits (Softmax is applied in CrossEntropyLoss)


class MLPChoiceModel(nn.Module):
    def __init__(self, embedding_dim=16, num_classes=10, hidden_dim=64):
        super().__init__()
        # An MLP (Multi-Layer Perceptron) allows for non-linear decision boundaries
        # in the embedding space.
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Regularization to prevent overfitting
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z):
        return self.net(z)
