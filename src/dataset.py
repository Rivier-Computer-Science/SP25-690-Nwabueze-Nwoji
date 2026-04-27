from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("data/raw", transform=transform)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

images, labels = next(iter(loader))

print("Image batch shape:", images.shape)
print("Labels:", labels)
print("Classes:", dataset.classes)