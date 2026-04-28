import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import BasicCNN

# ── data ────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder("data/raw", transform=transform)

# Use only the first 64 samples — enough to test overfitting
small_dataset = Subset(dataset, indices=range(64))
loader = DataLoader(small_dataset, batch_size=8, shuffle=True)

# ── model / loss / optimizer ─────────────────────────────────────────────────
model = BasicCNN(num_classes=24)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ── training loop ────────────────────────────────────────────────────────────
EPOCHS = 5

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    correct = 0

    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)          # forward pass
        loss = criterion(outputs, labels)  # compute cross-entropy loss
        loss.backward()                  # backprop
        optimizer.step()                 # update weights

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(small_dataset) * 100
    print(f"Epoch {epoch}/{EPOCHS}  loss: {avg_loss:.4f}  acc: {accuracy:.1f}%")

# Goal: loss should approach 0 and acc → 100% by epoch 5 (overfit confirmed)
