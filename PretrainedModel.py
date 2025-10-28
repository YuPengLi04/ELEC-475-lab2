import torch.nn as nn
from torchvision.models import alexnet, vgg16, AlexNet_Weights, VGG16_Weights

def _replace_head(m: nn.Module, in_features: int) -> nn.Sequential:
    """Return a small regression head ending in 2 outputs."""
    return nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.25),
        nn.Linear(1024, 2)          # (u, v)
    )

def build_model(backbone: str = "alexnet",
                pretrained: bool = True,
                freeze_backbone: bool = False) -> nn.Module:
    """
    backbone: 'alexnet' | 'vgg16'
    pretrained: load torchvision ImageNet weights
    freeze_backbone: if True, freeze all features (train head only)
    """
    backbone = backbone.lower()
    if backbone == "alexnet":
        weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        m = alexnet(weights=weights)
        # replace classifier[-1]
        in_features = m.classifier[-1].in_features   # 4096
        m.classifier[-1] = nn.Linear(in_features, 2)
        # (optional) beefier head:
        # m.classifier = nn.Sequential(*list(m.classifier[:-1]), nn.Linear(in_features, 2))
    elif backbone == "vgg16":
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        m = vgg16(weights=weights)
        in_features = m.classifier[-1].in_features   # 4096
        m.classifier[-1] = nn.Linear(in_features, 2)
    else:
        raise ValueError("backbone must be 'alexnet' or 'vgg16'")

    if freeze_backbone:
        # freeze all except the final layer we just inserted
        for p in m.parameters():
            p.requires_grad = False
        # unfreeze the last linear we replaced
        for p in m.classifier[-1].parameters():
            p.requires_grad = True

    return m
