import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50

from .mapping import FixedPixelMapping


class PLMResNet50(nn.Module):
    def __init__(self, initialization="scratch", pretrained_path=None):
        super().__init__()
        self.mapping = FixedPixelMapping()
        self.classifier = resnet50(weights=None)
        if initialization == "imagenet":
            if not pretrained_path:
                raise ValueError("pretrained_path is required for ImageNet initialization")
            state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            self.classifier.load_state_dict(state, strict=True)
        elif initialization != "scratch":
            raise ValueError(f"Unknown initialization: {initialization}")
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, 2)

    def forward(self, uint8_images):
        return self.classifier(self.mapping(uint8_images))


class LoRALinear(nn.Module):
    """Frozen linear projection with a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scaling = float(alpha) / rank
        self.dropout = nn.Dropout(float(dropout))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs):
        base = F.linear(inputs, self.weight, self.bias)
        update = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return base + self.scaling * update


def _replace_module(root: nn.Module, name: str, replacement: nn.Module):
    parent = root
    components = name.split(".")
    for component in components[:-1]:
        parent = getattr(parent, component)
    setattr(parent, components[-1], replacement)


def apply_attention_lora(model, rank=8, alpha=16.0, dropout=0.0,
                         start_layer=1, end_layer=24):
    """Wrap q/k/v/o projections in the selected 1-based transformer layers."""
    candidates = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.rsplit(".", 1)[-1] not in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            continue
        parts = name.split(".")
        layer_index = next(
            (int(parts[index + 1]) for index, part in enumerate(parts[:-1])
             if part in {"layer", "layers"} and parts[index + 1].isdigit()),
            None,
        )
        if layer_index is not None and start_layer <= layer_index + 1 <= end_layer:
            candidates.append((name, module))
    if not candidates:
        raise RuntimeError("No DINOv3 attention projections were found for LoRA")
    for name, module in candidates:
        _replace_module(model, name, LoRALinear(module, rank, alpha, dropout))
    return [name for name, _ in candidates]


class PLMDINOv3LoRA(nn.Module):
    """Fixed PLM input mapping followed by DINOv3-LoRA and a linear head."""

    def __init__(self, pretrained_path, rank=8, alpha=16.0, dropout=0.0,
                 start_layer=1, end_layer=24, gradient_checkpointing=True,
                 train_block_mlps=False, pool_patch_tokens=False):
        super().__init__()
        if not pretrained_path:
            raise ValueError("pretrained_path is required for DINOv3 LoRA")
        from transformers import AutoModel

        self.mapping = FixedPixelMapping()
        self.backbone = AutoModel.from_pretrained(pretrained_path, local_files_only=True)
        self.backbone.requires_grad_(False)
        self.lora_modules = apply_attention_lora(
            self.backbone, rank, alpha, dropout, start_layer, end_layer,
        )
        self.train_block_mlps = bool(train_block_mlps)
        self.pool_patch_tokens = bool(pool_patch_tokens)
        if self.train_block_mlps:
            for name, parameter in self.backbone.named_parameters():
                if ".mlp." in name:
                    parameter.requires_grad = True
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        feature_dim = self.backbone.config.hidden_size * (2 if self.pool_patch_tokens else 1)
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, uint8_images):
        mapped = self.mapping(uint8_images)
        hidden = self.backbone(pixel_values=mapped).last_hidden_state
        features = hidden[:, 0]
        if self.pool_patch_tokens:
            patch_start = 1 + int(self.backbone.config.num_register_tokens)
            pooled_patches = hidden[:, patch_start:].mean(dim=1)
            features = torch.cat((features, pooled_patches), dim=1)
        return self.classifier(features)

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


def build_model(config):
    model_name = config.get("model", "resnet50")
    if model_name == "resnet50":
        return PLMResNet50(config["initialization"], config.get("pretrained_path"))
    if model_name == "dinov3_lora":
        lora = config["lora"]
        return PLMDINOv3LoRA(
            pretrained_path=config["pretrained_path"],
            rank=lora["rank"], alpha=lora["alpha"], dropout=lora["dropout"],
            start_layer=lora["start_layer"], end_layer=lora["end_layer"],
            gradient_checkpointing=lora.get("gradient_checkpointing", True),
            train_block_mlps=lora.get("train_block_mlps", False),
            pool_patch_tokens=config.get("pool_patch_tokens", False),
        )
    raise ValueError(f"Unknown model: {model_name}")
