#!/usr/bin/env python3
"""Evaluate a reproducible degraded subset and optionally save its images."""

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

import torch
from PIL import Image

from plm.data import PLMTransform, discover_split
from plm.degradation import DegradationConfig, DegradationPipeline, PROFILES
from plm.model import PLMResNet50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--class-name", choices=("ai", "nature"), default="ai")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--degradation-profile", choices=PROFILES, default="sr")
    parser.add_argument("--degradation-level", type=int, choices=range(6), default=5)
    parser.add_argument("--degradation-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    image_output = output / "images"
    output.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        image_output.mkdir(exist_ok=True)

    label = 1 if args.class_name == "ai" else 0
    candidates = [row for row in discover_split(args.data_root, "val") if row["label"] == label]
    if args.count > len(candidates):
        raise ValueError(f"Requested {args.count}, but only {len(candidates)} images are available")
    selected = random.Random(args.sample_seed).sample(candidates, args.count)
    selected.sort(key=lambda row: row["path"])

    degradation_config = DegradationConfig(
        args.degradation_profile, args.degradation_level, args.degradation_seed, True
    )
    degradation = DegradationPipeline(degradation_config)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["configuration"]
    model = PLMResNet50(config["initialization"], config.get("pretrained_path"))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.cuda().eval().requires_grad_(False)
    crop = PLMTransform(config["crop_size"], train=False)

    rows = []
    for start in range(0, len(selected), args.batch_size):
        batch_rows = selected[start:start + args.batch_size]
        tensors, metadata = [], []
        for position, row in enumerate(batch_rows, start=start):
            source_path = Path(args.data_root) / row["path"]
            with Image.open(source_path) as source:
                degraded = degradation(source, key=row["path"])
            saved_path = None
            if args.save_images:
                saved_path = image_output / f"{position:03d}_{source_path.stem}.png"
                degraded.save(saved_path)
            tensors.append(crop(degraded))
            metadata.append((row, saved_path))
        images = torch.stack(tensors).cuda(non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=config["precision"] == "bf16"):
            probabilities = model(images).float().softmax(1)[:, 1].cpu().tolist()
        for (row, saved_path), probability in zip(metadata, probabilities):
            rows.append({
                "source_path": row["path"],
                "degraded_path": str(saved_path.relative_to(output)) if saved_path else None,
                "label": label,
                "fake_probability": probability,
                "prediction": int(probability >= 0.5),
            })

    probabilities = [row["fake_probability"] for row in rows]
    predicted_fake = sum(row["prediction"] for row in rows)
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_epoch": checkpoint["epoch"] + 1,
        "data_root": str(Path(args.data_root).resolve()),
        "split": "val",
        "class_name": args.class_name,
        "count": len(rows),
        "sample_seed": args.sample_seed,
        "degradation": degradation_config.__dict__,
        "crop": f"center {config['crop_size']}x{config['crop_size']}",
        "mean_fake_probability": statistics.fmean(probabilities),
        "median_fake_probability": statistics.median(probabilities),
        "min_fake_probability": min(probabilities),
        "max_fake_probability": max(probabilities),
        "classified_fake": predicted_fake,
        "classified_real": len(rows) - predicted_fake,
        "fake_accuracy": predicted_fake / len(rows) if label == 1 else None,
    }
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
