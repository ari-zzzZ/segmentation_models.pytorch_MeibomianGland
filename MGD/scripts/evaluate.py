"""Re-evaluate a trained checkpoint offline on its held-out test split."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mgd.core import GlandDataset, evaluate, make_model, read_rows
import torch
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; existing directories are refused",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    rows = read_rows(config, verify_hashes=True)
    ds = GlandDataset([r for r in rows if r["split"] == "test"], config)
    loader = DataLoader(ds, batch_size=config["training"]["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics, records = evaluate(
        model, loader, config, device, args.output / "predictions"
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "best_epoch": checkpoint["epoch"],
        "metrics": metrics,
        "per_image": records,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
