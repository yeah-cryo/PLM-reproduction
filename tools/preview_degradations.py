#!/usr/bin/env python3
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from plm.degradation import DegradationConfig, DegradationPipeline, PROFILES


def main():
    parser = argparse.ArgumentParser(description="Preview all levels of a degradation profile")
    parser.add_argument("image")
    parser.add_argument("--profile", choices=PROFILES, default="sr")
    parser.add_argument("--output", default="degradation_preview.jpg")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with Image.open(args.image) as source:
        source = source.convert("RGB")
    thumb_size = (256, 256)
    panels = []
    for level in range(6):
        pipeline = DegradationPipeline(DegradationConfig(args.profile, level, args.seed))
        degraded = pipeline(source, key=Path(args.image).as_posix())
        degraded.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (256, 288), "white")
        panel.paste(degraded, ((256 - degraded.width) // 2, (256 - degraded.height) // 2))
        ImageDraw.Draw(panel).text((8, 266), f"{args.profile} level {level}", fill="black")
        panels.append(panel)
    grid = Image.new("RGB", (3 * 256, 2 * 288), "white")
    for index, panel in enumerate(panels):
        grid.paste(panel, ((index % 3) * 256, (index // 3) * 288))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output, quality=95)
    print(Path(args.output).resolve())


if __name__ == "__main__":
    main()
