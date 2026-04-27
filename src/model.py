import torch
import torch.nn as nn


class BasicCNN(nn.Module):
    """
    Minimal CNN to test overfitting on a small batch.
    Architecture: Conv→ReLU→MaxPool → Conv→ReLU→MaxPool → FC→Softmax
    Input: (B, 3, 64, 64)  — matches dataset.py resize
    """

    def __init__(self, num_classes: int = 26):
        super().__init__()

        # Block 1: 3→16 feature maps, 3×3 kernel; 64×64 → 32×32 after pool
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Block 2: 16→32 feature maps; 32×32 → 16×16 after pool
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Fully connected: flatten 32×16×16 → num_classes
        self.fc = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)   # flatten
        return self.fc(x)            # raw logits (use CrossEntropyLoss, not Softmax here)


# ── quick sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    model = BasicCNN(num_classes=26)
    dummy = torch.randn(8, 3, 64, 64)   # same batch size as dataset.py
    out = model(dummy)
    print("Output shape:", out.shape)   # expect (8, 26)
