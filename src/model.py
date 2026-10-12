from __future__ import annotations

from typing import Callable, Optional, Type

import torch
import torch.nn as nn


class SimpleCytoCNN(nn.Module):
    """
    Резервна проста CNN-архітектура.

    У фінальній версії основною є SE-ResNet18 Attention. SimpleCytoCNN
    залишена тільки як fallback для старих експериментів/checkpoint-файлів.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ChannelSEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.

    Важлива деталь для БКР-патча: блок ініціалізується identity-сумісно.
    Це дозволяє завантажити вже навчений ResNet18 checkpoint як backbone без
    руйнування результатів класифікації. Після fine-tuning SE-блоки можуть
    навчитися підсилювати/послаблювати канали ознак.
    """

    def __init__(self, channels: int, reduction: int = 16, attention_strength: float = 0.20) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_channels, channels)
        self.sigmoid = nn.Sigmoid()
        self.attention_strength = attention_strength

        # Initial gate: 1.0 + strength * (sigmoid(0) - 0.5) = 1.0
        nn.init.kaiming_uniform_(self.fc1.weight, a=1.0)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.shape
        pooled = self.avg_pool(x).view(batch_size, channels)
        weights = self.fc1(pooled)
        weights = self.relu(weights)
        weights = self.fc2(weights)
        weights = self.sigmoid(weights)
        scale = 1.0 + self.attention_strength * (weights - 0.5)
        scale = scale.view(batch_size, channels, 1, 1)
        return x * scale


class BasicBlock(nn.Module):
    """
    Minimal BasicBlock compatible with torchvision ResNet18 state_dict keys.

    Імена шарів навмисно збігаються з torchvision: conv1, bn1, relu,
    conv2, bn2, downsample. Це дозволяє завантажувати checkpoint, навчений
    через torchvision ResNet18 у Google Colab.
    """

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        attention_factory: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.attention = attention_factory(planes) if attention_factory is not None else nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEBasicBlock(BasicBlock):
    """ResNet18 BasicBlock з Squeeze-and-Excitation Attention."""

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample,
            attention_factory=lambda channels: ChannelSEBlock(channels),
        )


class LocalResNet(nn.Module):
    """Minimal torchvision-compatible ResNet implementation for inference."""

    def __init__(
        self,
        block: Type[BasicBlock],
        layers: list[int],
        num_classes: int,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers: list[nn.Module] = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def build_local_resnet18(num_classes: int) -> nn.Module:
    return LocalResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def build_se_resnet18(num_classes: int) -> nn.Module:
    return LocalResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def build_resnet18(num_classes: int) -> nn.Module:
    """
    Створює ResNet18, сумісний із torchvision checkpoint.

    Для inference використовується локальна реалізація, щоб застосунок не залежав
    від compiled ops torchvision. Це зменшує ризик помилок середовища на Windows
    і в Python-середовищах із несумісними версіями torch/torchvision.
    """

    return build_local_resnet18(num_classes=num_classes)


def normalize_model_name(model_name: str) -> str:
    return model_name.strip().lower().replace("-", "_").replace(" ", "_")


def build_model(num_classes: int, model_name: str = "se_resnet18") -> nn.Module:
    normalized_name = normalize_model_name(model_name)

    if normalized_name in {"resnet18", "resnet_18", "baseline_resnet18"}:
        return build_resnet18(num_classes=num_classes)

    if normalized_name in {
        "se_resnet18",
        "se_resnet_18",
        "attention_resnet18",
        "resnet18_attention",
        "cnn_attention",
    }:
        return build_se_resnet18(num_classes=num_classes)

    if normalized_name in {"simple_cnn", "simplecytocnn", "cnn"}:
        return SimpleCytoCNN(num_classes=num_classes)

    raise ValueError(
        f"Непідтримувана архітектура моделі: {model_name}. "
        "Підтримуються: se_resnet18, resnet18, simple_cnn."
    )


def model_has_attention(model_name: str) -> bool:
    normalized_name = normalize_model_name(model_name)
    return normalized_name in {
        "se_resnet18",
        "se_resnet_18",
        "attention_resnet18",
        "resnet18_attention",
        "cnn_attention",
    }


def get_model_display_name(model_name: str) -> str:
    normalized_name = normalize_model_name(model_name)

    if normalized_name in {"se_resnet18", "se_resnet_18", "attention_resnet18", "resnet18_attention", "cnn_attention"}:
        return "SE-ResNet18 Attention"

    if normalized_name in {"resnet18", "resnet_18", "baseline_resnet18"}:
        return "ResNet18 baseline"

    if normalized_name in {"simple_cnn", "simplecytocnn", "cnn"}:
        return "SimpleCytoCNN"

    return model_name


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """
    Повертає останній convolutional block для Grad-CAM.

    Для ResNet18 / SE-ResNet18 це layer4[-1].
    Для SimpleCytoCNN це останній convolutional блок features.
    """

    if hasattr(model, "layer4"):
        return model.layer4[-1]

    if isinstance(model, SimpleCytoCNN):
        return model.features[-3]

    raise ValueError(
        "Не вдалося визначити target layer для Grad-CAM. "
        "Перевір архітектуру моделі."
    )
