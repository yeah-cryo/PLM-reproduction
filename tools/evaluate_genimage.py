#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from plm.data import (GenImageDataset, PLMTransform, collate_valid, discover_split,
                      seed_worker)
from plm.degradation import DegradationConfig, DegradationPipeline, PROFILES
from plm.metrics import binary_metrics
from plm.model import build_model


GENERATORS = {
    "SDv1.4": "stable_diffusion_v_1_4",
    "Midjourney": "Midjourney",
    "SDv1.5": "stable_diffusion_v_1_5",
    "ADM": "ADM",
    "GLIDE": "glide",
    "Wukong": "wukong",
    "VQDM": "VQDM",
    "BigGAN": "BigGAN",
}


def find_dataset_root(genimage_root, folder):
    base = Path(genimage_root) / folder
    candidates = [base] if (base / "val/nature").is_dir() else []
    candidates.extend(p for p in base.iterdir()
                      if (p / "val/nature").is_dir() and (p / "val/ai").is_dir())
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise ValueError(f"Expected one validation root below {base}, found {candidates}")
    return candidates[0]


@torch.no_grad()
def evaluate(model, loader, device, precision):
    labels, probabilities = [], []
    for images, targets in loader:
        if images is None:
            continue
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            logits = model(images)
        probabilities.extend(logits.float().softmax(1)[:, 1].cpu().tolist())
        labels.extend(targets.tolist())
    return binary_metrics(labels, probabilities)


def main():
    parser = argparse.ArgumentParser(description="Evaluate PLM on all official GenImage val splits")
    parser.add_argument("--checkpoint", default="outputs/genimage_sd14_fixed/final.pt")
    parser.add_argument("--data-root", default="/mnt/f/datasets/GenImage")
    parser.add_argument("--output", default="outputs/genimage_sd14_fixed/genimage_evaluation")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--degradation-profile", choices=PROFILES, default="clean",
                        help="Clean, atomic, or composed degradation profile")
    parser.add_argument("--degradation-level", type=int, choices=range(6), default=0,
                        metavar="{0..5}", help="0 is clean; 5 is extreme")
    parser.add_argument("--degradation-seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["configuration"]
    model = build_model(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.cuda().eval().requires_grad_(False)
    digest = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    results = []
    degradation_config = DegradationConfig(
        profile=args.degradation_profile,
        level=args.degradation_level,
        seed=args.degradation_seed,
        deterministic=True,
    )
    degradation = DegradationPipeline(degradation_config)
    for name, folder in GENERATORS.items():
        root = find_dataset_root(args.data_root, folder)
        rows = discover_split(root, "val")
        transform = PLMTransform(config["crop_size"], train=False, degradation=degradation)
        dataset = GenImageDataset(root, rows, transform,
                                  output / f"{name}.corrupt.jsonl", config["seed"], "skip")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker,
                            persistent_workers=args.workers > 0, collate_fn=collate_valid)
        row = {"generator": name, **evaluate(model, loader, torch.device("cuda"), config["precision"])}
        results.append(row)
        print(json.dumps(row), flush=True)
    mapping_config = config.get("mapping", {"type": "fixed"})
    if mapping_config.get("type", "fixed") == "smooth":
        mapping_description = (
            f"seeded smooth piecewise-linear mapping "
            f"(h={mapping_config.get('spacing', 16)}, seed={mapping_config.get('seed', 42)}, "
            f"per_channel={mapping_config.get('per_channel', True)})"
        )
    else:
        mapping_description = "Equation 4 fixed mapping"
    summary = {"checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": digest,
               "epoch": checkpoint["epoch"] + 1, "crop": f"center {config['crop_size']}x{config['crop_size']}",
               "mapping": mapping_description, "threshold": 0.5,
               "degradation": {"profile": degradation_config.profile,
                               "level": degradation_config.level,
                               "seed": degradation_config.seed},
               "results": results,
               "mean_accuracy": sum(x["accuracy"] for x in results) / len(results),
               "mean_average_precision": sum(x["average_precision"] for x in results) / len(results)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({"mean_accuracy": summary["mean_accuracy"],
                      "mean_average_precision": summary["mean_average_precision"]}), flush=True)


if __name__ == "__main__":
    main()
