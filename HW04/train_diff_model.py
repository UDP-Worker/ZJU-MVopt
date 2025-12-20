import argparse
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on diff features (.npy).")
    parser.add_argument("--features", default="features.npy", help="Path to features .npy.")
    parser.add_argument("--labels", default="labels.npy", help="Path to labels .npy.")
    parser.add_argument("--out-dir", default="runs", help="Output directory for logs/checkpoints.")
    parser.add_argument("--run-name", default="", help="Run name (default: timestamp).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--no-balanced-sampler",
        action="store_true",
        help="Disable weighted sampler (enabled by default).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="Batches between training log lines (default: 20).",
    )
    parser.add_argument("--input-size", type=int, default=0, help="Resize input to square size.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable input normalization.")
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split(labels, val_ratio, test_ratio, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    indices = np.arange(labels.shape[0])
    train_idx = []
    val_idx = []
    test_idx = []
    for cls in np.unique(labels):
        cls_idx = indices[labels == cls]
        rng.shuffle(cls_idx)
        n_total = len(cls_idx)
        n_test = int(round(n_total * test_ratio))
        n_val = int(round(n_total * val_ratio))
        n_test = min(n_test, n_total)
        n_val = min(n_val, n_total - n_test)
        test_idx.extend(cls_idx[:n_test])
        val_idx.extend(cls_idx[n_test : n_test + n_val])
        train_idx.extend(cls_idx[n_test + n_val :])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


class DiffDataset(Dataset):
    def __init__(self, features_path, labels_path, normalize=True, input_size=0):
        self.features = np.load(features_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        self.normalize = normalize
        self.input_size = input_size
        self.dtype = self.features.dtype

    def __len__(self):
        return int(self.labels.shape[0])

    def _normalize(self, x):
        if not self.normalize:
            return x
        if self.dtype == np.uint8:
            return (x - 127.5) / 127.5
        if self.dtype == np.int16:
            return x / 255.0
        return x / 255.0

    def __getitem__(self, idx):
        x = self.features[idx]
        y = int(self.labels[idx])
        if x.ndim == 4 and x.shape[-1] == 1:
            x = x[..., 0]
        x = x.astype(np.float32, copy=False)
        x = self._normalize(x)
        if x.ndim == 3:
            x = torch.from_numpy(x)
        else:
            x = torch.from_numpy(x)
        if self.input_size and self.input_size > 0:
            x = x.unsqueeze(0)
            x = torch.nn.functional.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
            x = x.squeeze(0)
        return x, torch.tensor(y, dtype=torch.float32)


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x).squeeze(1)


def compute_metrics(logits, targets):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return acc, prec, rec, f1


def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())
    if not all_logits:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    acc, prec, rec, f1 = compute_metrics(logits, targets)
    avg_loss = total_loss / float(len(targets))
    return avg_loss, acc, prec, rec, f1


def main():
    args = parse_args()
    set_seed(args.seed)

    if not os.path.isfile(args.features):
        print(f"error: features file not found: {args.features}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.labels):
        print(f"error: labels file not found: {args.labels}", file=sys.stderr)
        return 1

    dataset = DiffDataset(
        args.features,
        args.labels,
        normalize=not args.no_normalize,
        input_size=args.input_size,
    )
    labels = np.asarray(dataset.labels)
    train_idx, val_idx, test_idx = stratified_split(
        labels, args.val_split, args.test_split, args.seed
    )
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        print("error: split produced empty subset. Adjust split ratios.", file=sys.stderr)
        return 1

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    train_labels = labels[train_idx]
    pos_count = int((train_labels == 1).sum())
    neg_count = int((train_labels == 0).sum())
    if pos_count == 0 or neg_count == 0:
        print("error: training split lacks one class.", file=sys.stderr)
        return 1

    sampler = None
    if not args.no_balanced_sampler:
        weights = np.where(train_labels == 1, neg_count / pos_count, 1.0).astype(np.float32)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight = torch.tensor([neg_count / float(pos_count)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = SimpleCNN(in_channels=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=out_dir)
    writer.add_text("config", str(vars(args)))
    writer.add_text("class_counts", f"pos={pos_count}, neg={neg_count}")

    best_f1 = -1.0
    best_path = os.path.join(out_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = len(train_loader)
        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            if args.log_interval > 0 and (step % args.log_interval == 0 or step == num_batches):
                avg_loss = running_loss / float(step * args.batch_size)
                print(
                    f"epoch {epoch}/{args.epochs} "
                    f"step {step}/{num_batches} "
                    f"train_loss={avg_loss:.4f}"
                )
        train_loss = running_loss / float(len(train_set))
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
            model, val_loader, device, loss_fn
        )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("metrics/val_acc", val_acc, epoch)
        writer.add_scalar("metrics/val_prec", val_prec, epoch)
        writer.add_scalar("metrics/val_rec", val_rec, epoch)
        writer.add_scalar("metrics/val_f1", val_f1, epoch)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)

        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_f1={val_f1:.4f}"
        )

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
        model, test_loader, device, loss_fn
    )
    writer.add_scalar("loss/test", test_loss)
    writer.add_scalar("metrics/test_acc", test_acc)
    writer.add_scalar("metrics/test_prec", test_prec)
    writer.add_scalar("metrics/test_rec", test_rec)
    writer.add_scalar("metrics/test_f1", test_f1)
    writer.close()

    print(
        f"test_loss={test_loss:.4f} "
        f"test_acc={test_acc:.4f} "
        f"test_prec={test_prec:.4f} "
        f"test_rec={test_rec:.4f} "
        f"test_f1={test_f1:.4f}"
    )
    print(f"best model saved to: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
