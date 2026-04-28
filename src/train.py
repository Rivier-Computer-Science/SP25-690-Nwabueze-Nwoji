import torch
import torch.nn as nn

from dataset import get_splits
from model import BasicCNN

#  data 
train_loader, val_loader, test_loader, classes = get_splits(batch_size=32)

#  model / loss / optimizer 
model = BasicCNN(num_classes=len(classes))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#  helpers
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


#  training loop 
EPOCHS = 20

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss,   val_acc   = run_epoch(val_loader,   train=False)

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"train loss: {train_loss:.4f}  acc: {train_acc:.1f}% | "
          f"val loss: {val_loss:.4f}  acc: {val_acc:.1f}%")
