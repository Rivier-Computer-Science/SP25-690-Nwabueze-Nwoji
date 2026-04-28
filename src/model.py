import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class BasicCNN(nn.Module):
    """
    Deeper CNN for ASL classification.
    Architecture:
        Conv->ReLU -> Conv->ReLU -> Pool   (block1: extract low-level features)
        Conv->ReLU -> Conv->ReLU -> Pool   (block2: extract high-level features)
        FC -> Dropout -> FC              (classifier head)
    Input: (B, 3, 64, 64)
    """

    def __init__(self, num_classes: int = 24, dropout: float = 0.5):
        super().__init__()

        # Block 1: two convs before pooling — 64x64 -> 32x32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Block 2: two convs before pooling — 32x32 -> 16x16
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Classifier: flatten 64x16x16 -> 256 -> num_classes
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


class ResNetASL(nn.Module):
    """
    Pretrained ResNet18 fine-tuned for ASL classification.
    All layers frozen except the final FC — only the classifier is retrained.
    Use this after BasicCNN plateaus.
    Input: (B, 3, 64, 64)
    """

    def __init__(self, num_classes: int = 24):
        super().__init__()

        # Load ResNet18 pretrained on ImageNet
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze all layers — we only want to retrain the final classifier
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace final FC (512 -> 1000) with one suited for ASL
        self.model.fc = nn.Linear(512, num_classes)

        # When ready to squeeze more accuracy out of it, unfreeze the last block by adding:
        """ for param in self.model.layer4.parameters():
        #     param.requires_grad = True"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# quick sanity check
if __name__ == "__main__":
    dummy = torch.randn(8, 3, 64, 64)
    cnn = BasicCNN(num_classes=24)
    resnet = ResNetASL(num_classes=24)
    print("BasicCNN output:", cnn(dummy).shape)     # expect (8, 24)
    print("ResNetASL output:", resnet(dummy).shape)  # expect (8, 24)
