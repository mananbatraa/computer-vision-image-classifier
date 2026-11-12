"""Transfer-learning training: head-only warm-up, then fine-tune the last blocks.

    python -m classifier.train                       # full EuroSAT, MobileNetV3-Large
    python -m classifier.train --limit-per-class 300 --epochs-head 1 --epochs-finetune 1   # quick run
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from .data import load_eurosat, make_loaders, pick_device, stratified_splits
from .model import build_model, count_trainable, freeze_backbone

ROOT = Path(__file__).resolve().parents[1]


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    preds, targets = [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            batch_pred = logits.argmax(1)
            correct += (batch_pred == labels).sum().item()
            seen += len(labels)
            preds.append(batch_pred.cpu())
            targets.append(labels.cpu())
    return total_loss / seen, correct / seen, torch.cat(preds).numpy(), torch.cat(targets).numpy()


def train_phase(name, model, loaders, device, epochs, lr, history, best):
    train_loader, val_loader, _ = loaders
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    print(f"\n[{name}] trainable parameters: {count_trainable(model):,}")
    for epoch in range(1, epochs + 1):
        start = time.time()
        tr_loss, tr_acc, *_ = run_epoch(model, train_loader, device, criterion, optimizer)
        va_loss, va_acc, *_ = run_epoch(model, val_loader, device, criterion)
        scheduler.step()
        history.append({"phase": name, "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "seconds": round(time.time() - start, 1)})
        print(f"  epoch {epoch}: train acc {tr_acc:.3f} · val acc {va_acc:.3f} "
              f"· {history[-1]['seconds']}s")
        if va_acc > best["val_acc"]:  # keep the weights from the best validation epoch
            best.update(val_acc=va_acc, state={k: v.detach().cpu().clone()
                                               for k, v in model.state_dict().items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", default="mobilenet_v3_large",
                        choices=["mobilenet_v3_small", "mobilenet_v3_large"])
    parser.add_argument("--size", type=int, default=128, help="input resolution")
    parser.add_argument("--epochs-head", type=int, default=2)
    parser.add_argument("--epochs-finetune", type=int, default=4)
    parser.add_argument("--finetune-blocks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = pick_device()
    dataset = load_eurosat()
    classes = dataset.classes
    splits = stratified_splits(dataset.targets, seed=args.seed, limit_per_class=args.limit_per_class)
    loaders = make_loaders(dataset, splits, size=args.size, batch_size=args.batch_size)
    print(f"device={device} · train/val/test = {[len(s) for s in splits]} · classes={len(classes)}")

    model = build_model(len(classes), args.arch).to(device)
    history, best = [], {"val_acc": -1.0, "state": None}
    started = time.time()

    # Phase 1: only the new head learns; the ImageNet features stay fixed.
    freeze_backbone(model, trainable_blocks=0)
    train_phase("head", model, loaders, device, args.epochs_head, 1e-3, history, best)
    # Phase 2: unfreeze the last blocks so features adapt to satellite imagery, at a lower LR.
    freeze_backbone(model, trainable_blocks=args.finetune_blocks)
    train_phase("finetune", model, loaders, device, args.epochs_finetune, 2e-4, history, best)

    model.load_state_dict(best["state"])
    _, test_acc, preds, targets = run_epoch(model, loaders[2], device, nn.CrossEntropyLoss())
    report = classification_report(targets, preds, target_names=classes, output_dict=True, digits=4)
    cm = confusion_matrix(targets, preds).tolist()
    worst = sorted(((classes[i], classes[j], cm[i][j]) for i in range(len(classes))
                    for j in range(len(classes)) if i != j and cm[i][j]), key=lambda t: -t[2])[:5]

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best["state"], "classes": classes, "arch": args.arch, "size": args.size},
               args.out / "eurosat_mobilenet.pt")
    metrics = {
        "dataset": "EuroSAT RGB", "arch": args.arch, "input_size": args.size, "device": str(device),
        "splits": {"train": len(splits[0]), "val": len(splits[1]), "test": len(splits[2])},
        "best_val_acc": best["val_acc"], "test_accuracy": test_acc,
        "test_macro_f1": report["macro avg"]["f1-score"],
        "per_class": {c: {k: round(report[c][k], 4) for k in ("precision", "recall", "f1-score")}
                      for c in classes},
        "top_confusions": [{"true": t, "predicted": p, "count": n} for t, p, n in worst],
        "confusion_matrix": {"labels": classes, "matrix": cm},
        "history": history,
        "training_minutes": round((time.time() - started) / 60, 1),
    }
    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nTest accuracy {test_acc:.4f} · macro F1 {metrics['test_macro_f1']:.4f} "
          f"· {metrics['training_minutes']} min")
    print("Most common confusions:", worst)


if __name__ == "__main__":
    main()
