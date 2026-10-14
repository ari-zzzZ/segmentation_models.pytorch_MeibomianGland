"""Shared full-image segmentation pipeline; paths are relative to MGD/."""

import csv
import hashlib
import json
import os
from pathlib import Path
import random

WORK = Path(__file__).resolve().parents[2]
REPO = WORK.parent
os.environ.setdefault("HF_HOME", str(REPO / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import albumentations as A
import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import Dataset
import yaml


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    height, width = config["data"]["image_size"]
    assert height % 32 == width % 32 == 0
    assert config["model"]["in_channels"] == 3 and config["model"]["classes"] == 1
    assert config["training"]["epochs"] > 0 and config["training"]["batch_size"] > 0
    return config


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def read_rows(config, verify_hashes=False):
    path = WORK / config["data"]["manifest"]
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    assert rows and len({r["id"] for r in rows}) == len(rows)
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    summary = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest() == summary["manifest_sha256"]
    ), "Manifest changed"
    for row in rows:
        for key in ["image", "gland_mask", "eyelid_mask"]:
            item = (WORK / row[key]).resolve()
            assert item.is_relative_to(WORK) and item.is_file(), f"Missing file: {item}"
            if verify_hashes:
                assert (
                    hashlib.sha256(item.read_bytes()).hexdigest()
                    == row[key + "_sha256"]
                ), f"Changed file: {item}"
    return rows


class GlandDataset(Dataset):
    def __init__(self, rows, config, augment=False):
        self.rows = rows
        self.height, self.width = config["data"]["image_size"]
        self.threshold = config["data"]["mask_threshold"]
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.augment = (
            A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Affine(
                        scale=(0.9, 1.1),
                        translate_percent=(-0.03, 0.03),
                        rotate=(-10, 10),
                        interpolation=cv2.INTER_LINEAR,
                        mask_interpolation=cv2.INTER_NEAREST,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=0,
                        fill_mask=0,
                        p=0.5,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15, contrast_limit=0.15, p=0.4
                    ),
                ],
                seed=config["seed"],
            )
            if augment
            else None
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(WORK / row["image"]) as image:
            image = np.array(image.convert("RGB"))
        with Image.open(WORK / row["gland_mask"]) as mask:
            mask = (np.array(mask.convert("L")) >= self.threshold).astype(np.uint8)
        image = cv2.resize(
            image, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        mask = cv2.resize(
            mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST
        )
        if self.augment:
            transformed = self.augment(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
        image = (image.astype(np.float32) / 255.0 - self.mean) / self.std
        return (
            torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            torch.from_numpy(np.ascontiguousarray(mask[None].astype(np.float32))),
            row["id"],
        )


def make_model(config, pretrained=True):
    model = config["model"]
    return smp.create_model(
        arch=model["architecture"],
        encoder_name=model["encoder"],
        encoder_weights=model["encoder_weights"] if pretrained else None,
        in_channels=model["in_channels"],
        classes=model["classes"],
    )


def segmentation_loss(logits, target):
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target
    ) + smp.losses.DiceLoss(mode="binary", from_logits=True)(logits, target)


def scores(tp, fp, fn):
    tp, fp, fn = int(tp), int(fp), int(fn)
    return {
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
    }


def save_prediction(output, row, pred, target):
    output.mkdir(parents=True, exist_ok=True)
    height, width = pred.shape
    with Image.open(WORK / row["image"]) as original:
        image = np.array(original.convert("RGB").resize((width, height)))
        original_size = original.size
    Image.fromarray(pred.astype(np.uint8) * 255).resize(
        original_size, Image.Resampling.NEAREST
    ).save(output / (row["id"] + "_mask.png"))
    # White: image; green: ground truth; red: prediction. Panels labelled by filename README.
    gt = image.copy()
    gt[target] = (0.5 * gt[target] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    overlay = image.copy()
    overlay[pred] = (0.5 * overlay[pred] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    Image.fromarray(np.concatenate([image, gt, overlay], axis=1)).save(
        output / (row["id"] + "_comparison.jpg")
    )


@torch.inference_mode()
def evaluate(model, loader, config, device, output=None):
    model.eval()
    totals = np.zeros(3, dtype=np.int64)
    records = []
    loss_sum = 0.0
    by_id = {r["id"]: r for r in loader.dataset.rows}
    for image, target, ids in loader:
        image, target = image.to(device), target.to(device)
        with torch.autocast(
            device_type=device.type,
            enabled=config["training"]["amp"] and device.type == "cuda",
        ):
            logits = model(image)
            loss = segmentation_loss(logits, target)
        assert torch.isfinite(loss) and torch.isfinite(logits).all(), (
            "Nonfinite evaluation"
        )
        loss_sum += loss.item() * len(ids)
        prediction = (
            (logits.sigmoid() >= config["training"]["prediction_threshold"])
            .cpu()
            .numpy()[:, 0]
        )
        truth = target.cpu().numpy()[:, 0].astype(bool)
        for name, pred, gt in zip(ids, prediction, truth):
            tp, fp, fn = (
                int((pred & gt).sum()),
                int((pred & ~gt).sum()),
                int((~pred & gt).sum()),
            )
            totals += [tp, fp, fn]
            records.append(
                {"id": name, **scores(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn}
            )
            if output:
                save_prediction(output, by_id[name], pred, gt)
    metrics = {
        "loss": loss_sum / len(records),
        "mean_dice": float(np.mean([r["dice"] for r in records])),
        "mean_iou": float(np.mean([r["iou"] for r in records])),
        **{"micro_" + key: value for key, value in scores(*totals).items()},
        "count": len(records),
        "metric_resolution_hw": config["data"]["image_size"],
    }
    return metrics, records
