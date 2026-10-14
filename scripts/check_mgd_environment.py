"""Check the local SMP installation, MGD1k pairing, and a CUDA training step."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

import segmentation_models_pytorch as smp


def main():
    root = Path(__file__).resolve().parents[1]
    assert (
        Path(smp.__file__)
        .resolve()
        .is_relative_to(root / "segmentation_models_pytorch")
    )
    assert torch.cuda.is_available(), "CUDA is unavailable"
    dataset = root / "data" / "Expore MGD1k Dataset"
    images = {p.stem: p for p in (dataset / "Original Images").iterdir() if p.is_file()}
    glands = {
        p.stem: p for p in (dataset / "Meibomian Gland Labels").iterdir() if p.is_file()
    }
    eyelids = {p.stem: p for p in (dataset / "Eyelid Lebels").iterdir() if p.is_file()}
    assert images and images.keys() == glands.keys() == eyelids.keys(), (
        "Unmatched image/mask names"
    )
    stem = min(images)
    with Image.open(images[stem]) as image:
        image_size = image.size
        image_array = np.array(image.convert("L"))
    with Image.open(glands[stem]) as mask:
        assert mask.size == image_size
        mask_mode = mask.mode
        mask_array = np.array(mask)
        mask_values = np.unique(mask_array).tolist()
    # Synthetic binary target: do not assume the annotation's foreground encoding.
    resized = cv2.resize(image_array, (256, 256)).astype(np.float32) / 255.0
    inputs = torch.from_numpy(resized)[None, None].to("cuda")
    targets = torch.randint(0, 2, inputs.shape, device="cuda").float()
    model = smp.Unet(
        encoder_name="resnet18", encoder_weights=None, in_channels=1, classes=1
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(inputs)
        loss = smp.losses.DiceLoss(mode="binary", from_logits=True)(logits, targets)
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets
        )
    assert logits.shape == inputs.shape and torch.isfinite(loss)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "smp": smp.__version__,
                "smp_path": smp.__file__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_memory_gib": round(
                    torch.cuda.get_device_properties(0).total_memory / 2**30, 2
                ),
                "image_count": len(images),
                "gland_mask_count": len(glands),
                "eyelid_mask_count": len(eyelids),
                "sample_image_size": image_size,
                "sample_gland_mask_mode": mask_mode,
                "sample_gland_mask_values": mask_values,
                "output_shape": list(logits.shape),
                "loss": loss.item(),
                "cuda_amp_training_step": "passed (random initialization and synthetic target)",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
