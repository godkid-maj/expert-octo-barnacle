"""
train.py
--------
Fine-tunes a ResNet18 (pretrained on ImageNet) to classify pet breeds using
the Oxford-IIIT Pet dataset (37 classes, ~7,400 real photos of cats & dogs).
The dataset downloads automatically via torchvision — no manual data prep.

Swap in your own dataset by pointing `USE_CUSTOM_DATA=True` at an
ImageFolder-style directory (train/<class_name>/*.jpg, val/<class_name>/*.jpg)
— see `get_dataloaders()` below.

Usage:
    python train.py --epochs 8 --batch-size 32

Outputs:
    model/model.pth      -- fine-tuned weights
    model/classes.json   -- ordered list of class names (index -> label)
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

DATA_DIR = Path("data")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Set this to True and edit get_dataloaders() to train on your own images
# organized as data/train/<class>/*.jpg and data/val/<class>/*.jpg
USE_CUSTOM_DATA = False


def get_dataloaders(batch_size: int, img_size: int = 224):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    if USE_CUSTOM_DATA:
        train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tfms)
        val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_tfms)
        class_names = train_ds.classes
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        return train_loader, val_loader, class_names

    # Default: Oxford-IIIT Pet, auto-downloaded on first run
    full_train = datasets.OxfordIIITPet(
        root=str(DATA_DIR), split="trainval", target_types="category",
        download=True, transform=train_tfms,
    )
    class_names = full_train.classes

    # carve out a validation split from trainval (90/10)
    n_val = int(0.1 * len(full_train))
    n_train = len(full_train) - n_val
    train_subset, val_subset_idx = random_split(
        full_train, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    # val subset needs eval-time transforms, not train-time augmentation, so
    # build it from a second instance of the dataset sharing the same split
    full_val = datasets.OxfordIIITPet(
        root=str(DATA_DIR), split="trainval", target_types="category",
        download=False, transform=val_tfms,
    )
    val_subset = torch.utils.data.Subset(full_val, val_subset_idx.indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, class_names


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    # Replace the final layer; new layer's params are trainable by default
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_n += images.size(0)
    torch.set_grad_enabled(True)

    return total_loss / total_n, total_correct / total_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--unfreeze-after", type=int, default=4,
                         help="epoch at which to unfreeze the backbone for fine-tuning")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = get_dataloaders(args.batch_size)
    print(f"{len(class_names)} classes, "
          f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val images")

    model = build_model(len(class_names), freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # Unfreeze the whole network partway through for full fine-tuning
        # at a lower learning rate, once the new head has stabilized.
        if epoch == args.unfreeze_after:
            print("Unfreezing backbone for full fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=args.lr / 10)

        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        dt = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} ({dt:.0f}s)  "
              f"train_loss={train_loss:.3f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "model.pth")
            with open(MODEL_DIR / "classes.json", "w") as f:
                json.dump(class_names, f)
            print(f"  -> saved new best model (val_acc={val_acc:.3f})")

    print(f"Done. Best val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to {MODEL_DIR / 'model.pth'}")


if __name__ == "__main__":
    main()
