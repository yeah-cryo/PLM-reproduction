import torch
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
