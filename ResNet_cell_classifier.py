import argparse
import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms


KEEP_CLASSES = ("EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL")


@dataclass(frozen=True)
class Example:
    image_path: Path
    label: int


class BloodCellsCsvDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        labels_csv: Path,
        class_names: Sequence[str] = KEEP_CLASSES,
        transform=None,
    ):
        self.images_dir = images_dir
        self.labels_csv = labels_csv
        self.class_names = tuple(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.transform = transform

        self.examples: List[Example] = self._load_examples()

    def _load_examples(self) -> List[Example]:
        if not self.labels_csv.exists():
            raise FileNotFoundError(f"labels_csv not found: {self.labels_csv}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"images_dir not found: {self.images_dir}")

        examples: List[Example] = []
        with self.labels_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                category = (row.get("Category") or "").strip()
                if category not in self.class_to_idx:
                    continue

                image_id_raw = row.get("Image")
                if image_id_raw is None:
                    raise ValueError("labels.csv missing required column: Image")
                image_id = int(str(image_id_raw).strip())
                fname = f"BloodImage_{image_id:05d}.jpg"
                image_path = self.images_dir / fname
                if not image_path.exists():
                    # Skip silently; some rows refer to images not present in JPEGImages.
                    continue

                examples.append(Example(image_path=image_path, label=self.class_to_idx[category]))

        if not examples:
            raise ValueError(
                "No usable examples found. "
                "Make sure labels.csv uses the same Image ids as JPEGImages."
            )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        img = Image.open(ex.image_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, ex.label


def stratified_split_indices(
    labels: Sequence[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    if train_ratio <= 0 or val_ratio < 0 or (train_ratio + val_ratio) >= 1:
        raise ValueError("ratios must satisfy: train_ratio > 0, val_ratio >= 0, train+val < 1")

    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[int(y)].append(i)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for _, idxs in sorted(by_class.items(), key=lambda kv: kv[0]):
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Ensure at least 1 sample ends up in test when possible.
        if n - (n_train + n_val) <= 0 and n >= 3:
            n_train = max(1, n_train - 1)

        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train : n_train + n_val])
        test_idx.extend(idxs[n_train + n_val :])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def stratified_train_val_indices(
    labels: Sequence[int],
    train_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must satisfy: 0 < train_ratio < 1")

    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[int(y)].append(i)

    train_idx: List[int] = []
    val_idx: List[int] = []
    for _, idxs in sorted(by_class.items(), key=lambda kv: kv[0]):
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(n * train_ratio))
        # Ensure at least 1 sample in val when possible.
        if n - n_train <= 0 and n >= 2:
            n_train = max(1, n_train - 1)
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, n_classes: int):
    model.eval()
    correct = 0
    total = 0
    per_class_correct = torch.zeros(n_classes, dtype=torch.long)
    per_class_total = torch.zeros(n_classes, dtype=torch.long)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()

        for c in range(n_classes):
            mask = y == c
            per_class_total[c] += mask.sum().cpu()
            per_class_correct[c] += (pred[mask] == c).sum().cpu()

    acc = correct / max(1, total)
    per_class_acc = (per_class_correct.float() / per_class_total.clamp_min(1).float()).tolist()
    return acc, per_class_acc


def build_resnet(model_name: str, n_classes: int, pretrained: bool) -> nn.Module:
    name = model_name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    elif name == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError("model_name must be one of: resnet18, resnet34, resnet50")

    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, n_classes)
    return m


def main():
    ap = argparse.ArgumentParser(description="4-class single-label blood cell classifier (ResNet).")
    ap.add_argument("--data-root", type=str, default="dataset-2master", help="Path to dataset-2master.")
    ap.add_argument(
        "--dataset",
        type=str,
        default="auto",
        help="auto|csv|imagefolder. auto: prefers dataset2-style images/TRAIN+TEST if present.",
    )
    ap.add_argument("--model", type=str, default="resnet18", help="resnet18|resnet34|resnet50")
    ap.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained weights.")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--train-ratio", type=float, default=0.75)
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Only used for csv dataset. For imagefolder dataset, val is (1-train_ratio) split from TRAIN.",
    )
    ap.add_argument("--seed", type=int, default=42)
    # Default to 0 for maximum compatibility (some environments block torch shared-memory helpers).
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    csv_images_dir = data_root / "JPEGImages"
    labels_csv = data_root / "labels.csv"
    folder_train_dir = data_root / "images" / "TRAIN"
    folder_test_dir = data_root / "images" / "TEST"

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    dataset_mode = args.dataset.lower()
    if dataset_mode not in ("auto", "csv", "imagefolder"):
        raise ValueError("--dataset must be one of: auto, csv, imagefolder")

    use_imagefolder = dataset_mode == "imagefolder" or (
        dataset_mode == "auto" and folder_train_dir.exists() and folder_test_dir.exists()
    )

    if use_imagefolder:
        full_train = datasets.ImageFolder(root=str(folder_train_dir), transform=train_tf)
        full_val = datasets.ImageFolder(root=str(folder_train_dir), transform=eval_tf)
        test_ds = datasets.ImageFolder(root=str(folder_test_dir), transform=eval_tf)

        train_idx, val_idx = stratified_train_val_indices(
            labels=full_train.targets, train_ratio=args.train_ratio, seed=args.seed
        )
        train_ds = Subset(full_train, train_idx)
        val_ds = Subset(full_val, val_idx)

        class_names = tuple(full_train.classes)
        n_classes = len(class_names)

        def subset_class_counts_imagefolder(ds: Subset) -> Dict[str, int]:
            ctr = Counter()
            for i in ds.indices:
                y = int(full_train.targets[i])
                ctr[class_names[y]] += 1
            return dict(ctr)

        print("dataset_mode: imagefolder")
        print("classes:", class_names)
        print(
            "split_sizes:",
            {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        )
        print("train_counts:", subset_class_counts_imagefolder(train_ds))
        print("val_counts:", subset_class_counts_imagefolder(val_ds))
        print("test_counts:", dict(Counter(class_names[y] for y in test_ds.targets)))
    else:
        base_ds = BloodCellsCsvDataset(images_dir=csv_images_dir, labels_csv=labels_csv, transform=None)
        labels = [ex.label for ex in base_ds.examples]
        class_names = base_ds.class_names
        n_classes = len(class_names)

        train_idx, val_idx, test_idx = stratified_split_indices(
            labels=labels, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
        )

        # Use different transforms for train vs eval.
        train_ds = Subset(
            BloodCellsCsvDataset(images_dir=csv_images_dir, labels_csv=labels_csv, transform=train_tf), train_idx
        )
        val_ds = Subset(
            BloodCellsCsvDataset(images_dir=csv_images_dir, labels_csv=labels_csv, transform=eval_tf), val_idx
        )
        test_ds = Subset(
            BloodCellsCsvDataset(images_dir=csv_images_dir, labels_csv=labels_csv, transform=eval_tf), test_idx
        )

        def subset_class_counts_csv(ds: Subset) -> Dict[str, int]:
            ctr = Counter()
            for i in ds.indices:
                ctr[class_names[labels[i]]] += 1
            return dict(ctr)

        print("dataset_mode: csv")
        print("classes:", class_names)
        print("dataset_size:", len(base_ds))
        print("split_sizes:", {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)})
        print("train_counts:", subset_class_counts_csv(train_ds))
        print("val_counts:", subset_class_counts_csv(val_ds))
        print("test_counts:", subset_class_counts_csv(test_ds))

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("device:", device)

    model = build_resnet(args.model, n_classes=n_classes, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pin_memory = device.type in ("cuda", "mps")
    common_loader_kwargs = dict(num_workers=args.num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **common_loader_kwargs)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            seen += y.size(0)

        train_loss = running_loss / max(1, seen)
        val_acc, val_per_class = evaluate(model, val_loader, device=device, n_classes=n_classes)
        print(
            f"epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  "
            f"val_acc={val_acc:.4f}  val_per_class={dict(zip(class_names, map(lambda x: round(x,4), val_per_class)))}"
        )

    test_acc, test_per_class = evaluate(model, test_loader, device=device, n_classes=n_classes)
    print("test_acc:", round(test_acc, 4))
    print(
        "test_per_class:",
        dict(zip(class_names, map(lambda x: round(x, 4), test_per_class))),
    )


if __name__ == "__main__":
    main()
