#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

from plm.data import (CLASSES, GenImageDataset, PLMTransform, discover_split,
                      read_manifest, seed_worker, write_manifest)
from plm.degradation import (DegradationConfig, DegradationPipeline,
                             RandomLevelDegradationPipeline)
from plm.gpu_degradation import GPURandomSRDegradation
from plm.model import PLMResNet50


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(payload, path):
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description="Train fixed-mapping ResNet-50 on GenImage SD1.4")
    parser.add_argument("--config", default="configs/genimage_sd14_fixed.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--output")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-batches", type=int, help="Bounded smoke-test mode")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    if args.output:
        config["output"] = args.output
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if config["global_batch_size"] % config["micro_batch_size"]:
        raise ValueError("global_batch_size must be divisible by micro_batch_size")
    if config["optimizer"].lower() != "adam" or config["learning_rate_schedule"] != "constant":
        raise ValueError("The paper protocol uses Adam and the documented reproduction uses constant LR")

    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "configuration.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    seed_all(config["seed"])

    manifest = output / "manifests/train.jsonl"
    if not manifest.exists():
        rows = discover_split(config["data_root"], "train")
        counts = {label: sum(row["label"] == value for row in rows) for label, value in CLASSES.items()}
        expected = config.get("expected_images_per_class")
        if expected and any(count != expected for count in counts.values()):
            raise ValueError(f"Unexpected class counts: {counts}")
        write_manifest(rows, manifest)
    rows = read_manifest(manifest)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()

    degradation = None
    gpu_degradation = None
    degradation_after_crop = False
    degradation_config = config.get("degradation")
    if degradation_config and degradation_config.get("enabled", True):
        if degradation_config.get("device", "cpu") == "cuda":
            if degradation_config["profile"] != "sr" or "levels" not in degradation_config:
                raise ValueError("CUDA degradation currently supports random-level SR only")
            gpu_degradation = GPURandomSRDegradation(degradation_config["levels"])
        elif "levels" in degradation_config:
            degradation = RandomLevelDegradationPipeline(
                profile=degradation_config["profile"],
                levels=degradation_config["levels"],
                seed=degradation_config.get("seed", config["seed"]),
                deterministic=degradation_config.get("deterministic", False),
            )
        else:
            degradation = DegradationPipeline(DegradationConfig(
                profile=degradation_config["profile"],
                level=degradation_config["level"],
                seed=degradation_config.get("seed", config["seed"]),
                deterministic=degradation_config.get("deterministic", False),
            ))
        degradation_after_crop = degradation_config.get("apply_after_crop", False)
    transform = PLMTransform(
        config["crop_size"], train=True, degradation=degradation,
        degradation_after_crop=degradation_after_crop,
    )
    dataset = GenImageDataset(config["data_root"], rows, transform,
                              output / "corrupt_train.jsonl", config["seed"])
    loader_generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(dataset, batch_size=config["global_batch_size"], shuffle=True,
                        generator=loader_generator, num_workers=config["workers"],
                        pin_memory=True, drop_last=True, worker_init_fn=seed_worker,
                        persistent_workers=config["workers"] > 0)
    device = torch.device("cuda")
    if gpu_degradation is not None:
        gpu_degradation = gpu_degradation.to(device)
    model = PLMResNet50(config["initialization"], config.get("pretrained_path")).to(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=config["learning_rate"],
                                 betas=tuple(config["betas"]), weight_decay=config["weight_decay"])
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["manifest_sha256"] != manifest_hash:
            raise ValueError("Training manifest differs from the resume checkpoint")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        loader_generator.set_state(checkpoint["loader_generator_state"])
        restore_rng(checkpoint["rng_state"])
        start_epoch = checkpoint["epoch"] + 1

    accumulation = config["global_batch_size"] // config["micro_batch_size"]
    print(json.dumps({"images": len(dataset), "epochs": config["epochs"],
                      "batches_per_epoch": len(loader), "global_batch_size": config["global_batch_size"],
                      "micro_batch_size": config["micro_batch_size"], "accumulation": accumulation,
                      "initialization": config["initialization"], "precision": config["precision"],
                      "degradation": degradation_config}), flush=True)
    metrics_path = output / "metrics.jsonl"
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        if gpu_degradation is not None:
            gpu_degradation.reset_stats()
        started = time.monotonic()
        loss_sum = correct = seen = 0
        for batch_index, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if gpu_degradation is not None:
                images = gpu_degradation(images)
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(labels), config["micro_batch_size"]):
                micro_images = images[offset:offset + config["micro_batch_size"]]
                micro_labels = labels[offset:offset + config["micro_batch_size"]]
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=config["precision"] == "bf16"):
                    logits = model(micro_images)
                    loss = F.cross_entropy(logits, micro_labels)
                (loss / accumulation).backward()
                loss_sum += loss.item() * len(micro_labels)
                correct += (logits.argmax(1) == micro_labels).sum().item()
                seen += len(micro_labels)
            optimizer.step()
            if (batch_index + 1) % config["log_interval"] == 0:
                print(json.dumps({"epoch": epoch + 1, "batch": batch_index + 1,
                                  "batches": len(loader), "loss": loss_sum / seen,
                                  "accuracy": correct / seen,
                                  "seconds": time.monotonic() - started}), flush=True)
            if args.max_batches and batch_index + 1 >= args.max_batches:
                break

        row = {"epoch": epoch + 1, "loss": loss_sum / seen, "accuracy": correct / seen,
               "images": seen, "learning_rate": optimizer.param_groups[0]["lr"],
               "seconds": time.monotonic() - started}
        if gpu_degradation is not None:
            row["degradation_level_counts"] = gpu_degradation.stats()
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "epoch": epoch, "configuration": config, "label_mapping": CLASSES,
                   "manifest_sha256": manifest_hash, "rng_state": rng_state(),
                   "loader_generator_state": loader_generator.get_state(),
                   "paper_protocol": "fixed pixel mapping, random 128 crop, ResNet-50, Adam",
                   "degradation": degradation_config}
        atomic_save(payload, output / "latest.pt")
        if (epoch + 1) % config["checkpoint_interval"] == 0 or epoch + 1 == config["epochs"]:
            atomic_save(payload, output / f"epoch_{epoch + 1:03d}.pt")
        print(json.dumps(row), flush=True)
    if config["epochs"] > 0:
        atomic_save(payload, output / "final.pt")


if __name__ == "__main__":
    main()
