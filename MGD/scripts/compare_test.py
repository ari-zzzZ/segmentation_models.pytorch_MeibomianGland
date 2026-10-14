"""Render all held-out images, labels, and before/after fine-tuning predictions."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mgd.core import GlandDataset, WORK, make_model, read_rows, scores, seed_all
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import DataLoader


@torch.inference_mode()
def predict(model, loader, config, device):
    model.eval()
    results = {}
    for image, target, names in loader:
        with torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda" and config["training"]["amp"],
        ):
            logits = model(image.to(device))
        assert torch.isfinite(logits).all()
        masks = (
            (logits.sigmoid() >= config["training"]["prediction_threshold"])
            .cpu()
            .numpy()[:, 0]
        )
        for name, mask, truth in zip(names, masks, target.numpy()[:, 0].astype(bool)):
            metric = scores(
                (mask & truth).sum(), (mask & ~truth).sum(), (~mask & truth).sum()
            )
            results[name] = {"mask": mask, "truth": truth, **metric}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    # Initialization reconstruction requires exactly the same model implementation.
    from mgd import core

    assert (
        Path(core.__file__).read_bytes()
        == (run / "code" / "mgd" / "core.py").read_bytes()
    ), "Pipeline differs from training snapshot"
    snapshot_rows = list(
        csv.DictReader((run / "split.csv").open(encoding="utf-8", newline=""))
    )
    assert snapshot_rows == read_rows(config, verify_hashes=True)
    rows = [row for row in snapshot_rows if row["split"] == "test"]
    assert len(rows) == 15
    loader = DataLoader(
        GlandDataset(rows, config),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(config["seed"])
    model = make_model(config).to(device)
    before = predict(model, loader, config, device)
    model.load_state_dict(checkpoint["model"])
    after = predict(model, loader, config, device)
    original_metrics = json.loads(
        (run / "test_metrics.json").read_text(encoding="utf-8")
    )
    after_dice = float(np.mean([value["dice"] for value in after.values()]))
    after_iou = float(np.mean([value["iou"] for value in after.values()]))
    assert abs(after_dice - original_metrics["mean_dice"]) < 1e-6
    assert abs(after_iou - original_metrics["mean_iou"]) < 1e-6
    before_dice = float(np.mean([value["dice"] for value in before.values()]))
    before_iou = float(np.mean([value["iou"] for value in before.values()]))
    output = run / "comparison"
    output.mkdir(exist_ok=True)
    cell_w, cell_h, gutter, header, row_header = 640, 320, 12, 180, 46
    width = 5 * cell_w + 6 * gutter
    height = header + len(rows) * (cell_h + row_header + gutter) + gutter
    canvas = Image.new("RGB", (width, height), "#eef2f6")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = (
        ImageFont.truetype(str(font_path), 24)
        if font_path.exists()
        else ImageFont.load_default()
    )
    title_font = ImageFont.truetype(str(font_path), 36) if font_path.exists() else font
    draw.text(
        (gutter, 10),
        "MGD1k · 15 张测试图 · U-Net + ResNet18 · 微调前后对比",
        font=title_font,
        fill="#122237",
    )
    draw.text(
        (gutter, 65),
        f"平均 Dice：训练前 {before_dice:.4f} → 训练后 {after_dice:.4f}    平均 IoU：{before_iou:.4f} → {after_iou:.4f}    最佳轮次：{checkpoint['epoch']}",
        font=font,
        fill="#122237",
    )
    draw.text(
        (gutter, 100),
        "训练前：相同 seed=42 重建（预训练编码器 + 随机解码器）；阈值 0.5；指标分辨率 320×640；全部测试样本按清单顺序展示。",
        font=font,
        fill="#46556a",
    )
    titles = [
        "原始图像",
        "Ground truth（白色为腺体）",
        "训练前预测（重建初始化）",
        "训练后预测（best.pt）",
        "训练后叠加（绿色 GT / 红色预测）",
    ]
    for col, title in enumerate(titles):
        draw.text(
            (gutter + col * (cell_w + gutter), 140), title, font=font, fill="#122237"
        )
    records = []
    for index, row in enumerate(rows):
        name = row["id"]
        pre, post = before[name], after[name]
        with Image.open(WORK / row["image"]) as original:
            image = cv2.resize(
                np.array(original.convert("RGB")),
                (cell_w, cell_h),
                interpolation=cv2.INTER_AREA,
            )
        truth, pred = post["truth"], post["mask"]
        overlay = image.copy()
        # Exclusive errors are colored; overlap yellow shows agreement.
        for mask, color in [
            (truth & ~pred, [0, 255, 0]),
            (pred & ~truth, [255, 0, 0]),
            (truth & pred, [255, 230, 0]),
        ]:
            overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array(color)).astype(
                np.uint8
            )
        panels = [
            Image.fromarray(image),
            Image.fromarray(truth.astype(np.uint8) * 255).convert("RGB"),
            Image.fromarray(pre["mask"].astype(np.uint8) * 255).convert("RGB"),
            Image.fromarray(pred.astype(np.uint8) * 255).convert("RGB"),
            Image.fromarray(overlay),
        ]
        y = header + index * (cell_h + row_header + gutter)
        caption = f"{index + 1:02d}  {name}     Dice {pre['dice']:.3f} → {post['dice']:.3f}   |   IoU {pre['iou']:.3f} → {post['iou']:.3f}"
        draw.text((gutter, y + 8), caption, font=font, fill="#122237")
        for col, panel in enumerate(panels):
            canvas.paste(panel, (gutter + col * (cell_w + gutter), y + row_header))
        records.append(
            {
                "id": name,
                "before_dice": pre["dice"],
                "after_dice": post["dice"],
                "before_iou": pre["iou"],
                "after_iou": post["iou"],
            }
        )
    canvas.save(output / "test15_before_after.png")
    canvas.save(output / "test15_before_after.jpg", quality=95)
    canvas.crop((0, 0, width, header + 3 * (cell_h + row_header + gutter))).save(
        output / "first3_preview.jpg", quality=95
    )
    summary = {
        "run": str(run),
        "best_epoch": checkpoint["epoch"],
        "count": len(rows),
        "before": {"mean_dice": before_dice, "mean_iou": before_iou},
        "after": {"mean_dice": after_dice, "mean_iou": after_iou},
        "initialization": "reconstructed with original seed and pretrained encoder; not a saved initial checkpoint",
        "overlay_legend": "green=GT only; red=prediction only; yellow=overlap",
        "per_image": records,
    }
    (output / "comparison_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "per_image"},
            indent=2,
        )
    )
    print(f"Figure: {output / 'test15_before_after.png'}")


if __name__ == "__main__":
    main()
