"""Train a configured experiment, or check readiness without starting a run."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mgd.core import (
    GlandDataset,
    WORK,
    evaluate,
    load_config,
    make_model,
    read_rows,
    seed_all,
    segmentation_loss,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def loaders(config, rows):
    return {
        split: DataLoader(
            GlandDataset(
                [r for r in rows if r["split"] == split],
                config,
                augment=split == "train",
            ),
            batch_size=config["training"]["batch_size"],
            shuffle=split == "train",
            num_workers=config["training"]["num_workers"],
            pin_memory=True,
            generator=torch.Generator().manual_seed(config["seed"]),
        )
        for split in ["train", "val", "test"]
    }


def training_step(model, batch, optimizer, scaler, config, device):
    model.train()
    image, target, _ = batch
    image, target = image.to(device), target.to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", enabled=config["training"]["amp"]):
        logits = model(image)
        loss = segmentation_loss(logits, target)
    assert torch.isfinite(loss), "Nonfinite training loss"
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config["training"]["gradient_clip"], error_if_nonfinite=True
    )
    scaler.step(optimizer)
    scaler.update()
    return loss.item(), float(norm)


def preflight(model, loader, optimizer, scaler, config, device):
    batch = next(iter(loader["train"]))
    image, target, ids = batch
    assert set(target.unique().tolist()) <= {0.0, 1.0}
    loss, norm = training_step(model, batch, optimizer, scaler, config, device)
    # Validate only two validation images, never test-set model performance.
    ds = GlandDataset(loader["val"].dataset.rows[:2], config)
    (WORK / "reports").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=WORK / "reports", prefix="preflight_") as temporary:
        temporary = Path(temporary)
        val_metrics, _ = evaluate(
            model,
            DataLoader(ds, batch_size=2),
            config,
            device,
            temporary / "predictions",
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
                "val_metrics": val_metrics,
            },
            temporary / "check.pt",
        )
        saved = torch.load(
            temporary / "check.pt", map_location=device, weights_only=True
        )
        model.load_state_dict(saved["model"])
        assert len(list((temporary / "predictions").glob("*_mask.png"))) == 2
    report = {
        "status": "passed",
        "torch": torch.__version__,
        "smp": smp.__version__,
        "gpu": torch.cuda.get_device_name(),
        "pretrained_weights": "loaded offline",
        "splits": {key: len(value.dataset) for key, value in loader.items()},
        "batch_shape": list(image.shape),
        "binary_target_values": target.unique().tolist(),
        "train_step_loss": loss,
        "gradient_norm": norm,
        "validation_forward_loss": val_metrics["loss"],
        "checkpoint_roundtrip": "passed",
        "prediction_export": "passed",
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "note": "One readiness optimizer step; no trained checkpoint saved. Not an effect evaluation.",
    }
    (WORK / "reports").mkdir(exist_ok=True)
    write_json(WORK / "reports" / "preflight.json", report)
    # Show actual transformed inputs and binary labels for human inspection.
    fig, axes = plt.subplots(2, 3, figsize=(15, 5))
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    for index in range(2):
        display = (image[index] * std + mean).permute(1, 2, 0).numpy().clip(0, 1)
        mask = target[index, 0].numpy()
        axes[index, 0].imshow(display)
        axes[index, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[index, 2].imshow(display)
        axes[index, 2].contour(mask, levels=[0.5], colors="lime", linewidths=0.5)
        for ax, title in zip(
            axes[index], ["Training input", "Gland target", "Alignment"]
        ):
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(WORK / "reports" / "training_input_preview.png", dpi=130)
    plt.close(fig)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=WORK / "configs" / "unet_resnet18_100.yaml"
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check files, offline weights and one GPU optimizer step; do not start training",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    seed_all(config["seed"])
    assert torch.cuda.is_available(), (
        "CUDA unavailable; select the project .venv interpreter"
    )
    device = torch.device("cuda")
    rows = read_rows(config, verify_hashes=True)
    loader = loaders(config, rows)
    model = make_model(config).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=training["amp"])
    if args.preflight:
        preflight(model, loader, optimizer, scaler, config, device)
        return
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=training["scheduler_patience"],
        factor=training["scheduler_factor"],
    )
    run = (
        WORK
        / "runs"
        / config["experiment"]
        / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6])
    )
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    manifest = WORK / config["data"]["manifest"]
    shutil.copy2(manifest, run / "split.csv")
    shutil.copy2(manifest.with_suffix(".json"), run / "split_summary.json")
    shutil.copytree(
        WORK / "src" / "mgd",
        run / "code" / "mgd",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(Path(__file__), run / "code" / "train.py")
    metadata = {
        "torch": torch.__version__,
        "smp": smp.__version__,
        "python": sys.version,
        "gpu": torch.cuda.get_device_name(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "split_unit": "image; patient-level independence not verified",
        "best_model_selection": "validation mean Dice",
        "prediction_threshold": training["prediction_threshold"],
    }
    write_json(run / "metadata.json", metadata)
    print(f"Run directory: {run}", flush=True)
    writer = SummaryWriter(str(run / "tensorboard"))
    history = []
    best = -1.0
    try:
        with (run / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            log = csv.DictWriter(
                stream,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_dice",
                    "val_iou",
                    "learning_rate",
                ],
            )
            log.writeheader()
            for epoch in range(1, training["epochs"] + 1):
                running = 0.0
                for batch in tqdm(
                    loader["train"], desc=f"Epoch {epoch}/{training['epochs']}"
                ):
                    loss, _ = training_step(
                        model, batch, optimizer, scaler, config, device
                    )
                    running += loss * len(batch[2])
                metrics, _ = evaluate(model, loader["val"], config, device)
                row = {
                    "epoch": epoch,
                    "train_loss": running / len(loader["train"].dataset),
                    "val_loss": metrics["loss"],
                    "val_dice": metrics["mean_dice"],
                    "val_iou": metrics["mean_iou"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                history.append(row)
                log.writerow(row)
                stream.flush()
                for key, value in row.items():
                    if key != "epoch":
                        writer.add_scalar(key, value, epoch)
                scheduler.step(metrics["mean_dice"])
                checkpoint = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "config": config,
                    "val_metrics": metrics,
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                torch.save(checkpoint, run / "last.pt")
                if metrics["mean_dice"] > best:
                    best = metrics["mean_dice"]
                    torch.save(checkpoint, run / "best.pt")
                print(
                    f"Epoch {epoch}: train loss={row['train_loss']:.4f}, val Dice={row['val_dice']:.4f}, val IoU={row['val_iou']:.4f}",
                    flush=True,
                )
        checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        # Test is accessed for metrics once, after all validation-based selection.
        test_metrics, records = evaluate(
            model, loader["test"], config, device, run / "test_predictions"
        )
        write_json(
            run / "test_metrics.json",
            {"best_epoch": checkpoint["epoch"], **test_metrics},
        )
        with (run / "test_per_image.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            output = csv.DictWriter(stream, fieldnames=list(records[0]))
            output.writeheader()
            output.writerows(records)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(
            [r["epoch"] for r in history],
            [r["train_loss"] for r in history],
            label="Train",
        )
        axes[0].plot(
            [r["epoch"] for r in history],
            [r["val_loss"] for r in history],
            label="Validation",
        )
        axes[0].set_title("Dice + BCE loss")
        axes[1].plot(
            [r["epoch"] for r in history],
            [r["val_dice"] for r in history],
            label="Dice",
        )
        axes[1].plot(
            [r["epoch"] for r in history], [r["val_iou"] for r in history], label="IoU"
        )
        axes[1].set_title("Validation metrics")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.legend()
        fig.tight_layout()
        fig.savefig(run / "learning_curves.png", dpi=150)
        plt.close(fig)
        print(
            f"Completed. Best epoch: {checkpoint['epoch']}; test mean Dice: {test_metrics['mean_dice']:.4f}; test mean IoU: {test_metrics['mean_iou']:.4f}"
        )
        print(f"Artifacts: {run}")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
