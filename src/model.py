import torch
import torch.nn as nn


class BasicCNN(nn.Module):
    """
    Deeper CNN for ASL classification.
    Architecture:
        Conv→ReLU → Conv→ReLU → Pool   (block1: extract low-level features)
        Conv→ReLU → Conv→ReLU → Pool   (block2: extract high-level features)
        FC → Dropout → FC              (classifier head)
    Input: (B, 3, 64, 64)
    """

    def __init__(self, num_classes: int = 24, dropout: float = 0.5):
        super().__init__()

        # Block 1: two convs before pooling — 64×64 → 32×32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Block 2: two convs before pooling — 32×32 → 16×16
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Classifier: flatten 64×16×16 → 256 → num_classes
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(dropout),        # reduces overfitting
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)      # flatten
        return self.classifier(x)       # raw logits


#  quick sanity check 
if __name__ == "__main__":
    model = BasicCNN(num_classes=24)
    dummy = torch.randn(8, 3, 64, 64)
    out = model(dummy)
    print("Output shape:", out.shape)   # expect (8, 24)
