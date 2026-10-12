from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Для одиночного inference у Streamlit надмірна кількість CPU threads може різко
# сповільнювати Grad-CAM на деяких Windows/Python середовищах.
try:
    if torch.get_num_threads() > 4:
        torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from .config import (
    ACTIVE_MODEL_NAME,
    ALLOW_RESNET18_CHECKPOINT_ATTENTION_ADAPTER,
    ATTENTION_ADAPTER_NOTE_UA,
    ATTENTION_EXPLANATION_UA,
    BASELINE_MODEL_NAME,
    BINARY_CLINICAL_LABELS,
    CLASS_DESCRIPTIONS_UA,
    CLASS_NAMES,
    CLINICAL_CONCLUSIONS,
    DATASET_NAME,
    GRADCAM_EXPLANATION_UA,
    MODEL_DISPLAY_NAME,
    MODEL_FAMILY,
    MODEL_PATH,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    RAW_CLASS_TO_DISPLAY,
    RECOMMENDED_ACTIONS,
    RISK_GROUPS,
    RISK_LEVELS,
    SYSTEM_LIMITATION_UA,
    TEST_ACCURACY,
    XAI_METHOD,
)
from .gradcam import GradCAM, heatmap_to_image, make_overlay
from .model import build_model, get_gradcam_target_layer, get_model_display_name, model_has_attention, normalize_model_name
from .preprocessing import image_to_tensor, load_image


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_load_safe(model_path: Path, device: torch.device) -> Any:
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


def _clean_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        cleaned_key = key

        if cleaned_key.startswith("module."):
            cleaned_key = cleaned_key.replace("module.", "", 1)

        if cleaned_key.startswith("model."):
            cleaned_key = cleaned_key.replace("model.", "", 1)

        cleaned[cleaned_key] = value

    return cleaned


def _state_dict_looks_like_resnet18(state_dict: dict[str, torch.Tensor]) -> bool:
    return (
        any(key.startswith("layer4.") for key in state_dict.keys())
        and any(key.startswith("fc.") for key in state_dict.keys())
    )


def _state_dict_looks_like_attention_resnet18(state_dict: dict[str, torch.Tensor]) -> bool:
    return (
        _state_dict_looks_like_resnet18(state_dict)
        and any(".attention." in key for key in state_dict.keys())
    )


def _state_dict_looks_like_simple_cnn(state_dict: dict[str, torch.Tensor]) -> bool:
    return (
        any(key.startswith("features.") for key in state_dict.keys())
        and any(key.startswith("classifier.") for key in state_dict.keys())
    )


def _extract_checkpoint_payload(payload: Any) -> tuple[dict[str, torch.Tensor], str, int]:
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        model_name = str(payload.get("model_name", BASELINE_MODEL_NAME))
        num_classes = int(payload.get("num_classes", len(CLASS_NAMES)))
        return state_dict, model_name, num_classes

    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        model_name = str(payload.get("model_name", BASELINE_MODEL_NAME))
        num_classes = int(payload.get("num_classes", len(CLASS_NAMES)))
        return state_dict, model_name, num_classes

    if isinstance(payload, dict):
        state_dict = payload

        if _state_dict_looks_like_attention_resnet18(state_dict):
            model_name = "se_resnet18"
        elif _state_dict_looks_like_resnet18(state_dict):
            model_name = BASELINE_MODEL_NAME
        elif _state_dict_looks_like_simple_cnn(state_dict):
            model_name = "simple_cnn"
        else:
            model_name = BASELINE_MODEL_NAME

        num_classes = len(CLASS_NAMES)
        return state_dict, model_name, num_classes

    raise ValueError("Непідтримуваний формат checkpoint моделі.")


def _extract_class_names(payload: Any, num_classes: int) -> list[str]:
    if isinstance(payload, dict) and "idx_to_class" in payload:
        raw_idx_to_class = payload["idx_to_class"]
        names: list[str] = []

        for idx in range(num_classes):
            raw_name = raw_idx_to_class.get(idx, raw_idx_to_class.get(str(idx), str(idx)))
            display_name = RAW_CLASS_TO_DISPLAY.get(raw_name, raw_name)
            names.append(display_name)

        return names

    if isinstance(payload, dict) and "class_to_idx" in payload:
        class_to_idx = payload["class_to_idx"]
        idx_to_class = {int(index): class_name for class_name, index in class_to_idx.items()}
        names = []

        for idx in range(num_classes):
            raw_name = idx_to_class.get(idx, str(idx))
            display_name = RAW_CLASS_TO_DISPLAY.get(raw_name, raw_name)
            names.append(display_name)

        return names

    return CLASS_NAMES[:num_classes]


def _filter_ignorable_missing_keys(
    missing_keys: list[str],
    *,
    effective_model_name: str,
    checkpoint_model_name: str,
    attention_adapter_enabled: bool,
) -> list[str]:
    """
    Старі локальні версії моделі могли мати зареєстрований target_layer.
    Якщо такі ключі відсутні у checkpoint, це не проблема.

    Для SE-ResNet18 можна завантажити ResNet18 checkpoint як backbone.
    У такому випадку параметри attention-блоків відсутні в checkpoint і
    ініціалізуються identity-сумісно у src/model.py.
    """

    ignored_prefixes = ["target_layer."]

    if attention_adapter_enabled:
        ignored_fragments = [".attention."]
    else:
        ignored_fragments = []

    filtered = []
    for key in missing_keys:
        if any(key.startswith(prefix) for prefix in ignored_prefixes):
            continue
        if any(fragment in key for fragment in ignored_fragments):
            continue
        filtered.append(key)

    return filtered


def _select_effective_model_name(checkpoint_model_name: str, state_dict: dict[str, torch.Tensor]) -> tuple[str, bool]:
    checkpoint_norm = normalize_model_name(checkpoint_model_name)
    active_norm = normalize_model_name(ACTIVE_MODEL_NAME)

    if _state_dict_looks_like_attention_resnet18(state_dict):
        return "se_resnet18", False

    if (
        active_norm in {"se_resnet18", "se_resnet_18", "attention_resnet18", "resnet18_attention", "cnn_attention"}
        and checkpoint_norm in {"resnet18", "resnet_18", "baseline_resnet18"}
        and _state_dict_looks_like_resnet18(state_dict)
        and ALLOW_RESNET18_CHECKPOINT_ATTENTION_ADAPTER
    ):
        return "se_resnet18", True

    return checkpoint_model_name, False


@lru_cache(maxsize=1)
def load_classifier(model_path: Path = MODEL_PATH) -> tuple[torch.nn.Module, list[str], dict[str, Any]]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint моделі не знайдено: {model_path}. "
            "Очікуваний файл: models/cyto_classifier.pth"
        )

    device = get_device()
    payload = _torch_load_safe(model_path, device)

    state_dict, checkpoint_model_name, num_classes = _extract_checkpoint_payload(payload)
    state_dict = _clean_state_dict_keys(state_dict)

    if _state_dict_looks_like_attention_resnet18(state_dict):
        checkpoint_model_name = "se_resnet18"
    elif _state_dict_looks_like_resnet18(state_dict):
        checkpoint_model_name = BASELINE_MODEL_NAME

    effective_model_name, attention_adapter_enabled = _select_effective_model_name(
        checkpoint_model_name=checkpoint_model_name,
        state_dict=state_dict,
    )

    model = build_model(num_classes=num_classes, model_name=effective_model_name).to(device)

    incompatible = model.load_state_dict(state_dict, strict=False)

    real_missing_keys = _filter_ignorable_missing_keys(
        list(incompatible.missing_keys),
        effective_model_name=effective_model_name,
        checkpoint_model_name=checkpoint_model_name,
        attention_adapter_enabled=attention_adapter_enabled,
    )
    unexpected_keys = list(incompatible.unexpected_keys)

    if real_missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint не відповідає архітектурі моделі.\n"
            f"Checkpoint model name: {checkpoint_model_name}\n"
            f"Effective model name: {effective_model_name}\n"
            f"Missing keys: {real_missing_keys[:20]}\n"
            f"Unexpected keys: {unexpected_keys[:20]}"
        )

    class_names = _extract_class_names(payload, num_classes)

    metadata = {
        "checkpoint_model_name": checkpoint_model_name,
        "effective_model_name": effective_model_name,
        "effective_model_display_name": get_model_display_name(effective_model_name),
        "configured_model_display_name": MODEL_DISPLAY_NAME,
        "model_family": MODEL_FAMILY,
        "dataset": DATASET_NAME,
        "xai_method": XAI_METHOD,
        "test_accuracy": TEST_ACCURACY,
        "num_classes": num_classes,
        "device": str(device),
        "checkpoint_path": str(model_path),
        "has_attention": model_has_attention(effective_model_name),
        "attention_adapter_enabled": attention_adapter_enabled,
        "attention_adapter_note": ATTENTION_ADAPTER_NOTE_UA if attention_adapter_enabled else "",
        "attention_explanation": ATTENTION_EXPLANATION_UA,
    }

    model.eval()
    return model, class_names, metadata


def build_recommended_action(predicted_class: str, confidence: float) -> str:
    action = RECOMMENDED_ACTIONS.get(
        predicted_class,
        "Рекомендована ручна верифікація спеціалістом.",
    )

    if confidence < 0.70:
        return (
            f"{action} Додатково: через низьку впевненість моделі результат потрібно "
            "обов’язково перевірити вручну, навіть якщо клас належить до групи норми."
        )

    if confidence < 0.85:
        return (
            f"{action} Додатково: через середню впевненість моделі бажано переглянути "
            "оригінальне зображення та Grad-CAM overlay."
        )

    return action


def build_decision_summary(predicted_class: str, confidence: float) -> str:
    conclusion = CLINICAL_CONCLUSIONS.get(predicted_class, "Потрібна ручна перевірка")
    risk_level = RISK_LEVELS.get(predicted_class, "Невизначений рівень ризику")
    confidence_percent = confidence * 100

    return (
        f"Система класифікувала зображення як {predicted_class}. "
        f"Клінічно-орієнтований висновок: {conclusion}. "
        f"Рівень ризику: {risk_level}. "
        f"Впевненість моделі: {confidence_percent:.2f}%."
    )


def predict_image(
    image_input: Any,
    filename_hint: str | None = None,
) -> dict[str, Any]:
    image = load_image(image_input)

    model, class_names, model_metadata = load_classifier()

    device = get_device()
    tensor = image_to_tensor(
        image,
        mean=NORMALIZATION_MEAN,
        std=NORMALIZATION_STD,
    ).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

    predicted_idx = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    target_layer = get_gradcam_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    try:
        heatmap = gradcam(tensor, target_class=predicted_idx)
    finally:
        gradcam.remove_hooks()

    overlay = make_overlay(image, heatmap)
    heatmap_img = heatmap_to_image(heatmap)

    top_indices = probabilities.argsort()[::-1][:3]
    top3 = [
        {
            "class": class_names[int(index)],
            "probability": float(probabilities[int(index)]),
        }
        for index in top_indices
    ]

    risk_group = RISK_GROUPS.get(predicted_class, "Потребує ручної верифікації")
    clinical_conclusion = CLINICAL_CONCLUSIONS.get(predicted_class, "Потрібна ручна перевірка")
    binary_clinical_label = BINARY_CLINICAL_LABELS.get(predicted_class, "manual_review")
    risk_level = RISK_LEVELS.get(predicted_class, "Невизначений рівень ризику")
    recommended_action = build_recommended_action(predicted_class, confidence)
    decision_summary = build_decision_summary(predicted_class, confidence)

    explanation = (
        f"{GRADCAM_EXPLANATION_UA} "
        "Результат є допоміжним і має бути перевірений спеціалістом."
    )

    return {
        "image": image,
        "filename_hint": filename_hint,
        "predicted_class": predicted_class,
        "predicted_idx": predicted_idx,
        "risk_group": risk_group,
        "clinical_conclusion": clinical_conclusion,
        "binary_clinical_label": binary_clinical_label,
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "decision_summary": decision_summary,
        "class_description": CLASS_DESCRIPTIONS_UA.get(
            predicted_class,
            "Опис класу відсутній.",
        ),
        "confidence": confidence,
        "top3": top3,
        "heatmap": heatmap,
        "heatmap_image": heatmap_img,
        "overlay": overlay,
        "explanation": explanation,
        "gradcam_explanation": GRADCAM_EXPLANATION_UA,
        "attention_explanation": ATTENTION_EXPLANATION_UA,
        "system_limitation": SYSTEM_LIMITATION_UA,
        "model_metadata": model_metadata,
    }
