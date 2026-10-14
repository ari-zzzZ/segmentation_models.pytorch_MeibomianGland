"""Create a reproducible, immutable 100-image subset and split manifests."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

WORK = Path(__file__).resolve().parents[1]
REPO = WORK.parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source = REPO / "data" / "Expore MGD1k Dataset"
    tag = f"mgd100_seed{args.seed}"
    manifest = WORK / "manifests" / f"{tag}.csv"
    if manifest.exists():
        print(f"Existing split retained: {manifest}")
        return
    images = sorted((source / "Original Images").glob("*.JPG"))
    assert len(images) == 1000, f"Expected 1000 images, found {len(images)}"
    strata = ["_".join(p.stem.split("_")[1:3]) for p in images]
    selected, _ = train_test_split(
        images, train_size=100, random_state=args.seed, stratify=strata
    )
    selected = sorted(selected)
    strata = ["_".join(p.stem.split("_")[1:3]) for p in selected]
    train, holdout = train_test_split(
        selected, train_size=70, random_state=args.seed, stratify=strata
    )
    holdout_strata = ["_".join(p.stem.split("_")[1:3]) for p in holdout]
    val, test = train_test_split(
        holdout, train_size=15, random_state=args.seed, stratify=holdout_strata
    )
    split_by_id = {
        p.stem: split
        for split, paths in [("train", train), ("val", val), ("test", test)]
        for p in paths
    }
    subset = WORK / "data" / "subsets" / tag
    rows = []
    for image in selected:
        row = {"id": image.stem, "split": split_by_id[image.stem]}
        row["stratum"] = "_".join(image.stem.split("_")[1:3])
        image_hash = sha256(image)
        row["image_sha256"] = image_hash
        with Image.open(image) as opened:
            size = opened.size
        row["width"], row["height"] = size
        for key, folder, filename in [
            ("image", "Original Images", image.name),
            ("gland_mask", "Meibomian Gland Labels", image.stem + ".png"),
            ("eyelid_mask", "Eyelid Lebels", image.stem + ".png"),
        ]:
            src = source / folder / filename
            with Image.open(src) as opened:
                assert opened.size == size, f"Size mismatch: {src}"
                opened.verify()
            dest = subset / key / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                assert sha256(dest) == sha256(src), f"Existing copy differs: {dest}"
            else:
                shutil.copy2(src, dest)
            row[key] = dest.relative_to(WORK).as_posix()
            if key != "image":
                row[key + "_sha256"] = sha256(dest)
        with Image.open(subset / "gland_mask" / (image.stem + ".png")) as mask:
            row["gland_foreground_fraction"] = float(
                (np.array(mask.convert("L")) >= 128).mean()
            )
        rows.append(row)
    # Prevent exact duplicate images from crossing partitions.
    hashes = {}
    for row in rows:
        previous = hashes.setdefault(row["image_sha256"], row["split"])
        assert previous == row["split"], "Duplicate content across splits"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "seed": args.seed,
        "source": str(source),
        "selected": len(rows),
        "counts": {
            split: sum(r["split"] == split for r in rows)
            for split in ["train", "val", "test"]
        },
        "strata": {
            stratum: {
                split: sum(
                    r["stratum"] == stratum and r["split"] == split for r in rows
                )
                for split in ["train", "val", "test"]
            }
            for stratum in sorted(set(strata))
        },
        "manifest_sha256": sha256(manifest),
        "mask_threshold": 128,
        "split_unit": "image (patient mapping unavailable)",
        "source_unchanged": True,
    }
    manifest.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
