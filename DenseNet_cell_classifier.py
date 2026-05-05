

"""
 DenseNet Blood Cell Classifier

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms, datasets
from torchvision.models import DenseNet121_Weights
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import random

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRAIN_DIR   = "/dataset2-master/dataset2-master/images/TRAIN"
TEST_DIR    = "/dataset2-master/dataset2-master/images/TEST"

IMG_SIZE    = 224       # DenseNet-121 native resolution (was 160)
BATCH_SIZE  = 32
NUM_EPOCHS  = 20        # More room before early stopping kicks in
HEAD_LR     = 3e-4      # Classifier head learning rate
BACKBONE_LR = 3e-5      # Backbone learning rate (10x smaller)
VAL_SPLIT   = 0.15
RANDOM_SEED = 42
PATIENCE    = 5         # More patience since we're unfreezing
SAVE_PATH   = "densenet_improved.pth"

DEVICE = torch.device(
    "cuda"  if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # Spatial augmentations
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),                       # Blood cells have no canonical orientation
    transforms.RandomRotation(45),                         # Full rotation invariance
    # Photometric augmentations — simulate staining variation
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    # Random erasing simulates out-of-focus / partial cells
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ─────────────────────────────────────────────
# DATASET SPLIT  (bug-fix: separate datasets, no transform leak)
# ─────────────────────────────────────────────
def make_splits(train_dir, val_fraction, seed):
    """
    Create train/val splits with *separate* transforms applied correctly.
    The original code mutated full_train.transform after random_split,
    which applied val_transform to ALL samples including the training ones.
    """
    full = datasets.ImageFolder(train_dir, transform=None)  # No transform yet
    n = len(full)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    val_size   = int(n * val_fraction)
    val_idx    = indices[:val_size]
    train_idx  = indices[val_size:]

    # Wrap with per-split transforms via a thin helper
    train_ds = TransformSubset(full, train_idx, train_transform)
    val_ds   = TransformSubset(full, val_idx,   val_transform)
    return train_ds, val_ds, full.classes


class TransformSubset(Subset):
    """Subset that applies its own transform, ignoring the parent dataset's."""
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset.imgs[self.indices[idx]]
        from PIL import Image
        img = Image.open(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]


# ─────────────────────────────────────────────
# MODEL  (differential LR: unfrozen backbone)
# ─────────────────────────────────────────────
def build_model(num_classes):
    model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

    # ── Unfreeze backbone entirely ──────────────────────────────────────
    # Differential LR (set below in optimizer) handles the risk of
    # catastrophic forgetting better than freezing.
    for param in model.features.parameters():
        param.requires_grad = True

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


def make_optimizer(model):
    """Separate param groups so backbone trains at 10x lower LR."""
    backbone_params = list(model.features.parameters())
    head_params     = list(model.classifier.parameters())
    return optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params,     "lr": HEAD_LR},
    ], weight_decay=1e-4)


# ─────────────────────────────────────────────
# TRAIN / EVAL
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping prevents exploding gradients when backbone is unfrozen
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="val", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        outputs = model(imgs)
        loss    = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Datasets
    train_ds, val_ds, classes = make_splits(TRAIN_DIR, VAL_SPLIT, RANDOM_SEED)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=val_transform)

    print(f"Classes : {classes}")
    print(f"Train   : {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # num_workers=2 is a safe cross-platform value; raise to 4 on Linux/CUDA
    nw = 2 if DEVICE.type != "cpu" else 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=nw, pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=nw, pin_memory=(DEVICE.type == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=nw, pin_memory=(DEVICE.type == "cuda"))

    # Class-weighted loss + label smoothing
    full_ds = datasets.ImageFolder(TRAIN_DIR)
    targets = [label for _, label in full_ds.samples]
    counts  = np.bincount(targets)
    weights = counts.sum() / (len(classes) * counts)
    weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    # Model, optimizer, scheduler
    model     = build_model(len(classes)).to(DEVICE)
    optimizer = make_optimizer(model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # Training loop
    best_val_acc     = 0.0
    epochs_no_improve = 0

    print("\nTraining...\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        current_lrs = [pg["lr"] for pg in optimizer.param_groups]
        print(f"Epoch {epoch:02d} | "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f} | "
              f"lr_backbone={current_lrs[0]:.2e}  lr_head={current_lrs[1]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            epochs_no_improve = 0
            print("  ✓ Saved best model")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping after {epoch} epochs (no improvement for {PATIENCE})")
            break

    # Final evaluation
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    _, test_acc, preds, truths = evaluate(model, test_loader, criterion)

    print(f"\nTest accuracy: {test_acc:.4f}\n")
    print(classification_report(truths, preds, target_names=classes))
    print(confusion_matrix(truths, preds))


if __name__ == "__main__":
    main()
