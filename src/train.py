import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset import get_splits
from model import BasicCNN, ResNetASL

# ── data ──────────────────────────────────────────────────────────────────────
train_loader, val_loader, test_loader, classes = get_splits(batch_size=32)

# ── model / loss / optimizer ──────────────────────────────────────────────────
# To switch models, uncomment the one you want to use:
# model = BasicCNN(num_classes=len(classes))  # current
# model = ResNetASL(num_classes=len(classes))  # upgrade
model = BasicCNN(num_classes=len(classes))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ── helpers ───────────────────────────────────────────────────────────────────
def run_epoch(loader, train: bool):
    model.train() if train else model.eval()
    total_loss, correct = 0.0, 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

    n = len(loader.dataset)
    return total_loss / len(loader), correct / n * 100


# ── training loop ─────────────────────────────────────────────────────────────
EPOCHS = 20

# track metrics each epoch for plotting
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss,   val_acc   = run_epoch(val_loader,   train=False)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"train loss: {train_loss:.4f}  acc: {train_acc:.1f}% | "
          f"val loss: {val_loss:.4f}  acc: {val_acc:.1f}%")

# Save weights + model name so evaluate.py always loads the correct architecture
torch.save({"model": type(model).__name__, "state_dict": model.state_dict()}, "models/cnn.pth")
print("Model saved to models/cnn.pth")

# ── plots ─────────────────────────────────────────────────────────────────────
epochs = range(1, EPOCHS + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss curve — gap between train/val reveals overfitting
ax1.plot(epochs, history["train_loss"], label="Train")
ax1.plot(epochs, history["val_loss"],   label="Val")
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.legend()

# Accuracy curve
ax2.plot(epochs, history["train_acc"], label="Train")
ax2.plot(epochs, history["val_acc"],   label="Val")
ax2.set_title("Accuracy (%)")
ax2.set_xlabel("Epoch")
ax2.legend()

plt.tight_layout()
plt.savefig("models/training_curves.png", dpi=150)
plt.show()
print("Saved: models/training_curves.png")
