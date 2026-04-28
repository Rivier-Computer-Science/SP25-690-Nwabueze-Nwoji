from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Augmentation for training — helps generalize to real-world hand positions/lighting
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomRotation(10),                          # handle tilted hands
    transforms.RandomAffine(0, translate=(0.1, 0.1)),       # handle off-center hands
    transforms.ColorJitter(brightness=0.2, contrast=0.2),   # lighting robustness
    transforms.ToTensor(),
])

# No augmentation for val/test — evaluate on clean images only
eval_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


def get_splits(data_dir: str = "data/raw", batch_size: int = 32):
    """Returns train, val, test DataLoaders with 70/15/15 split."""

    full = datasets.ImageFolder(data_dir, transform=train_transform)
    n = len(full)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val  # remainder goes to test

    train_set, val_set, test_set = random_split(full, [n_train, n_val, n_test])

    # Override transform for val/test so augmentation doesn't affect evaluation
    val_set.dataset.transform  = eval_transform
    test_set.dataset.transform = eval_transform

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size)
    test_loader  = DataLoader(test_set,  batch_size=batch_size)

    return train_loader, val_loader, test_loader, full.classes


if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_splits()
    print(f"Classes ({len(classes)}):", classes)
    images, labels = next(iter(train_loader))
    print("Train batch shape:", images.shape)
