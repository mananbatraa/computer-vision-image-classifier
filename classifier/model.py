"""MobileNetV3 with a new classification head for transfer learning."""
import torch
from torch import nn
from torchvision import models

ARCHS = {
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
}


def build_model(num_classes: int, arch: str = "mobilenet_v3_large", pretrained: bool = True) -> nn.Module:
    factory, weights = ARCHS[arch]
    model = factory(weights=weights if pretrained else None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)  # replace the 1000-class ImageNet head
    return model


def freeze_backbone(model: nn.Module, trainable_blocks: int = 0):
    """Freeze the convolutional feature extractor except its last `trainable_blocks` blocks.
    The classifier head always stays trainable."""
    for p in model.features.parameters():
        p.requires_grad = False
    if trainable_blocks:
        for block in list(model.features)[-trainable_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def predict_proba(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    model.eval()
    return torch.softmax(model(batch), dim=1)
