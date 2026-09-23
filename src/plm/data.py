import json
import random
import warnings
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


CLASSES = {"nature": 0, "ai": 1}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def discover_split(root, split):
    root = Path(root)
    rows = []
    for class_name, label in CLASSES.items():
        folder = root / split / class_name
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        rows.extend({"path": str(p.relative_to(root)), "label": label} for p in paths)
    return rows


def write_manifest(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_manifest(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pad_to(image, size):
    width, height = image.size
    dx, dy = max(0, size - width), max(0, size - height)
    if dx or dy:
        image = ImageOps.expand(image, (dx // 2, dy // 2, dx - dx // 2, dy - dy // 2), fill=0)
    return image


class PLMTransform:
    def __init__(self, size=128, train=False, degradation=None,
                 degradation_after_crop=False):
        self.size = int(size)
        self.train = bool(train)
        self.degradation = degradation
        self.degradation_after_crop = bool(degradation_after_crop)

    def __call__(self, image, key=None):
        image = image.convert("RGB")
        if self.degradation is not None and not self.degradation_after_crop:
            image = self.degradation(image, key=key)
        image = pad_to(image, self.size)
        width, height = image.size
        if self.train:
            top = random.randrange(height - self.size + 1)
            left = random.randrange(width - self.size + 1)
        else:
            top = (height - self.size) // 2
            left = (width - self.size) // 2
        image = TF.crop(image, top, left, self.size, self.size)
        if self.degradation is not None and self.degradation_after_crop:
            image = self.degradation(image, key=key)
        array = torch.from_numpy(__import__("numpy").array(image, dtype="uint8", copy=True))
        return array.permute(2, 0, 1).contiguous()


class GenImageDataset(Dataset):
    def __init__(self, root, rows, transform, corrupt_log=None, retry_seed=42,
                 corruption_policy="replace"):
        self.root = Path(root)
        self.rows = rows
        self.transform = transform
        self.corrupt_log = Path(corrupt_log) if corrupt_log else None
        self.retry_seed = int(retry_seed)
        self.corruption_policy = corruption_policy
        self.indices_by_label = {
            label: [index for index, row in enumerate(rows) if row["label"] == label]
            for label in CLASSES.values()
        }

    def __len__(self):
        return len(self.rows)

    def _load(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["path"]) as image:
            return self.transform(image, key=row["path"]), int(row["label"]), row["path"]

    def __getitem__(self, index):
        try:
            image, label, _ = self._load(index)
            return image, label
        except Exception as error:
            row = self.rows[index]
            warnings.warn(f"Unreadable image replaced: {row['path']}: {error}", RuntimeWarning)
            if self.corrupt_log:
                self.corrupt_log.parent.mkdir(parents=True, exist_ok=True)
                with self.corrupt_log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"path": row["path"], "error": str(error)}) + "\n")
            if self.corruption_policy == "skip":
                return None
            if self.corruption_policy != "replace":
                raise ValueError(f"Unknown corruption policy: {self.corruption_policy}")
            rng = random.Random(self.retry_seed + index)
            same_class = self.indices_by_label[row["label"]]
            for _ in range(32):
                try:
                    image, label, _ = self._load(rng.choice(same_class))
                    return image, label
                except Exception:
                    pass
            raise RuntimeError(f"Could not replace corrupt sample {row['path']}") from error


def collate_valid(batch):
    valid = [item for item in batch if item is not None]
    if not valid:
        return None, None
    images, labels = zip(*valid)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    __import__("numpy").random.seed(seed)
