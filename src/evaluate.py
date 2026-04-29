import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import get_splits
from model import BasicCNN, ResNetASL

# ── load data & model ─────────────────────────────────────────────────────────
_, _, test_loader, classes = get_splits(batch_size=32)

# Automatically builds the correct architecture from the saved checkpoint
checkpoint = torch.load("models/cnn.pth", weights_only=False)
model_name  = checkpoint["model"]
model = BasicCNN(num_classes=len(classes)) if model_name == "BasicCNN" else ResNetASL(num_classes=len(classes))
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print(f"Loaded: {model_name}")

# collect predictions on test set 
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        preds = model(images).argmax(1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

# confusion matrix 
cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(14, 12))
ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=False)
plt.title("Confusion Matrix — ASL Test Set")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=150)
plt.show()
print("Saved: models/confusion_matrix.png")

#  per-class accuracy 
# Shows exactly which letters the model struggles with (e.g. M vs N, U vs V)
print("\nPer-class accuracy:")
for i, cls in enumerate(classes):
    mask = [l == i for l in all_labels]
    correct = sum(p == l for p, l in zip(all_preds, all_labels) if l == i)
    total = sum(mask)
    print(f"  {cls}: {correct}/{total} ({100 * correct / total:.1f}%)")
