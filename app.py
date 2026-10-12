from __future__ import annotations

import base64
import html
import json
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from src.config import (
    ATTENTION_ADAPTER_NOTE_UA,
    ATTENTION_EXPLANATION_UA,
    BKR_COMPLIANCE_ITEMS,
    CLASS_NAMES,
    DATASET_NAME,
    GRADCAM_EXPLANATION_UA,
    MODEL_BACKBONE_NAME,
    MODEL_COMPARISON_PATH,
    MODEL_DISPLAY_NAME,
    MODEL_FAMILY,
    MODEL_PATH,
    REPORT_DIR,
    SAMPLE_DIR,
    SYSTEM_LIMITATION_UA,
    TEST_ACCURACY,
    TEST_CORRECT,
    TEST_TOTAL,
    XAI_METHOD,
)
from src.predict import predict_image


st.set_page_config(
    page_title="CytoXAI Workstation",
    layout="wide",
)

ALLOWED_SUFFIXES = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
UPLOAD_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]

MODEL_NAME = MODEL_DISPLAY_NAME
BASELINE_MODEL_NAME = MODEL_BACKBONE_NAME

CLASS_DISPLAY_NAMES = {
    "dyskeratotic": "Dyskeratotic",
    "koilocytotic": "Koilocytotic",
    "metaplastic": "Metaplastic",
    "parabasal": "Parabasal",
    "superficial_intermediate": "Superficial-Intermediate",
    "superficial-intermediate": "Superficial-Intermediate",
}


st.markdown(
    """
    <style>
        :root {
            --cyto-bg: #f6f8fb;
            --cyto-surface: #ffffff;
            --cyto-surface-soft: #f8fafc;
            --cyto-text: #0f172a;
            --cyto-muted: #475569;
            --cyto-border: #dbe4ef;
            --cyto-blue: #2563eb;
            --cyto-indigo: #1e40af;
            --cyto-green: #15803d;
            --cyto-amber: #b45309;
            --cyto-red: #b91c1c;
            --cyto-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
        }
        .stApp {
            background: radial-gradient(circle at top left, #eef6ff 0, #f8fafc 34%, #ffffff 100%);
        }
        .main .block-container {
            padding-top: 1.35rem;
            padding-bottom: 3rem;
            max-width: 1420px;
        }
        .cyto-hero {
            border: 1px solid rgba(37, 99, 235, 0.18);
            border-radius: 24px;
            padding: 24px 28px;
            margin-bottom: 18px;
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 64, 175, 0.92)),
                radial-gradient(circle at top right, rgba(96, 165, 250, 0.45), transparent 35%);
            box-shadow: 0 24px 52px rgba(15, 23, 42, 0.18);
            color: #ffffff;
        }
        .cyto-hero-kicker {
            color: #bfdbfe;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .cyto-hero-title {
            color: #ffffff;
            font-size: clamp(2rem, 4vw, 3rem);
            line-height: 1.06;
            font-weight: 900;
            margin: 0 0 10px 0;
        }
        .cyto-hero-text {
            color: #dbeafe;
            font-size: 1.03rem;
            line-height: 1.58;
            max-width: 980px;
            margin: 0;
        }
        .cyto-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 16px;
        }
        .cyto-chip {
            border: 1px solid rgba(191, 219, 254, 0.35);
            border-radius: 999px;
            padding: 7px 11px;
            color: #eff6ff;
            background: rgba(255, 255, 255, 0.10);
            font-size: 0.86rem;
            font-weight: 700;
        }
        .section-header {
            margin: 28px 0 14px 0;
            border-radius: 18px;
            padding: 15px 18px;
            background: linear-gradient(135deg, #0f172a, #1d4ed8);
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 14px 30px rgba(30, 64, 175, 0.18);
        }
        .section-header h2 {
            color: #ffffff !important;
            font-size: 1.28rem;
            line-height: 1.2;
            font-weight: 900;
            margin: 0;
        }
        .section-header p {
            color: #dbeafe !important;
            margin: 6px 0 0 0;
            line-height: 1.45;
            font-size: 0.94rem;
        }
        .clinical-card {
            min-height: 188px;
            border-radius: 20px;
            border: 1px solid var(--cyto-border);
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            padding: 18px 18px 16px 18px;
            box-shadow: var(--cyto-shadow);
            color: var(--cyto-text) !important;
            margin-bottom: 12px;
        }
        .clinical-card-title {
            color: #334155 !important;
            font-size: 0.78rem;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            margin-bottom: 10px;
        }
        .clinical-card-value {
            color: #0f172a !important;
            font-size: 1.2rem;
            font-weight: 900;
            line-height: 1.22;
            margin-bottom: 10px;
        }
        .clinical-card-text {
            color: #334155 !important;
            font-size: 0.95rem;
            line-height: 1.55;
            margin: 0;
        }
        .clinical-card.success { border-top: 5px solid var(--cyto-green); }
        .clinical-card.warning { border-top: 5px solid var(--cyto-amber); }
        .clinical-card.danger { border-top: 5px solid var(--cyto-red); }
        .clinical-card.info { border-top: 5px solid var(--cyto-blue); }
        .result-card {
            border-radius: 18px;
            border: 1px solid var(--cyto-border);
            background: #ffffff;
            padding: 16px;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
            margin-bottom: 12px;
        }
        .block-title {
            font-size: 0.82rem;
            font-weight: 900;
            color: #334155 !important;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .block-value {
            font-size: 1.05rem;
            font-weight: 800;
            color: #0f172a !important;
            margin-bottom: 0.25rem;
        }
        .block-text {
            color: #334155 !important;
            line-height: 1.55;
        }
        .small-muted {
            color: #475569 !important;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .xai-note {
            border-radius: 18px;
            border: 1px solid #dbeafe;
            background: linear-gradient(180deg, #eff6ff 0%, #ffffff 100%);
            padding: 16px 18px;
            box-shadow: 0 10px 26px rgba(37, 99, 235, 0.08);
            color: #172554 !important;
            margin-top: 12px;
        }
        .xai-note p, .xai-note div { color: #172554 !important; }
        .limitation-card {
            border-radius: 18px;
            border: 1px solid #fed7aa;
            background: linear-gradient(180deg, #fff7ed 0%, #ffffff 100%);
            padding: 16px 18px;
            box-shadow: 0 10px 26px rgba(180, 83, 9, 0.08);
            color: #7c2d12 !important;
            margin-top: 12px;
        }
        .limitation-card p, .limitation-card div { color: #7c2d12 !important; }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--cyto-border);
            border-radius: 18px;
            padding: 16px 18px;
            min-height: 108px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
            overflow: visible;
        }
        div[data-testid="stMetricLabel"] p {
            color: #475569 !important;
            font-weight: 800;
            font-size: 0.95rem !important;
            line-height: 1.2 !important;
            margin-bottom: 8px !important;
        }
        div[data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-weight: 900;
            overflow: visible !important;
        }
        div[data-testid="stMetricValue"] p {
            color: #0f172a !important;
            font-size: clamp(1.35rem, 2.3vw, 1.9rem) !important;
            line-height: 1.12 !important;
            font-weight: 900 !important;
            white-space: normal !important;
            overflow-wrap: anywhere !important;
            word-break: break-word !important;
        }
        /* Streamlit native alerts can become visually transparent on custom themes.
           Force solid, readable surfaces everywhere. */
        div[data-testid="stAlertContainer"] {
            background: #ffffff !important;
            border: 1px solid var(--cyto-border) !important;
            border-left: 6px solid var(--cyto-blue) !important;
            border-radius: 16px !important;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08) !important;
            color: #0f172a !important;
        }
        div[data-testid="stAlertContainer"] * {
            color: #0f172a !important;
        }
        div[data-testid="stAlertContainer"] p {
            color: #0f172a !important;
            font-weight: 650 !important;
            line-height: 1.48 !important;
        }
        div[data-testid="stAlertContainer"] [data-testid="stAlertContentWarning"] p {
            color: #78350f !important;
        }
        div[data-testid="stAlertContainer"] [data-testid="stAlertContentSuccess"] p {
            color: #14532d !important;
        }
        div[data-testid="stAlertContainer"] [data-testid="stAlertContentError"] p {
            color: #7f1d1d !important;
        }
        .cyto-status-card {
            border-radius: 16px;
            border: 1px solid var(--cyto-border);
            background: #ffffff;
            padding: 14px 16px;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.07);
            color: #0f172a !important;
            margin: 10px 0;
        }
        .cyto-status-card .status-title {
            color: #0f172a !important;
            font-size: 0.96rem;
            font-weight: 900;
            margin-bottom: 4px;
        }
        .cyto-status-card .status-text {
            color: #334155 !important;
            font-size: 0.95rem;
            line-height: 1.55;
            margin: 0;
        }
        .cyto-status-card.info { border-left: 6px solid var(--cyto-blue); background: linear-gradient(180deg, #eff6ff 0%, #ffffff 100%); }
        .cyto-status-card.success { border-left: 6px solid var(--cyto-green); background: linear-gradient(180deg, #f0fdf4 0%, #ffffff 100%); }
        .cyto-status-card.warning { border-left: 6px solid var(--cyto-amber); background: linear-gradient(180deg, #fffbeb 0%, #ffffff 100%); }
        .cyto-status-card.danger { border-left: 6px solid var(--cyto-red); background: linear-gradient(180deg, #fef2f2 0%, #ffffff 100%); }
        .cyto-detail-card {
            border-radius: 16px;
            border: 1px solid var(--cyto-border);
            background: #ffffff;
            padding: 14px 16px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.055);
            color: #0f172a !important;
            margin: 12px 0;
        }
        .cyto-detail-card .detail-title {
            color: #334155 !important;
            font-size: 0.8rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .cyto-detail-card .detail-text {
            color: #1e293b !important;
            font-size: 0.96rem;
            line-height: 1.55;
            margin: 0;
        }
        .cyto-meta-card {
            border-radius: 14px;
            border: 1px solid #c7d2fe;
            background: linear-gradient(135deg, #eef2ff 0%, #ffffff 100%);
            color: #1e1b4b !important;
            padding: 12px 14px;
            margin: 12px 0 4px 0;
            box-shadow: 0 8px 22px rgba(79, 70, 229, 0.08);
            font-size: 0.92rem;
            font-weight: 750;
            line-height: 1.45;
        }
        .cyto-meta-card * { color: #1e1b4b !important; }
        div[data-testid="stCaptionContainer"] {
            color: #334155 !important;
            background: #ffffff !important;
            border: 1px solid var(--cyto-border) !important;
            border-radius: 12px !important;
            padding: 8px 10px !important;
        }
        div[data-testid="stCaptionContainer"] p {
            color: #334155 !important;
        }
        .stDataFrame {
            border-radius: 16px;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_class_name(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def extract_expected_class(filename: Optional[str]) -> Optional[str]:
    if not filename:
        return None

    normalized = normalize_class_name(Path(filename).stem)

    aliases = sorted(
        CLASS_DISPLAY_NAMES.keys(),
        key=len,
        reverse=True,
    )

    for alias in aliases:
        if normalize_class_name(alias) in normalized:
            return CLASS_DISPLAY_NAMES[alias]

    return None


def classes_match(predicted_class: str, expected_class: Optional[str]) -> Optional[bool]:
    if expected_class is None:
        return None

    return normalize_class_name(predicted_class) == normalize_class_name(expected_class)


def get_confidence_status(confidence: float) -> tuple[str, str, str]:
    if confidence >= 0.85:
        return (
            "success",
            "Висока впевненість",
            "Модель сформувала впевнений результат класифікації. Результат все одно має розглядатися як допоміжний.",
        )

    if confidence >= 0.70:
        return (
            "warning",
            "Середня впевненість",
            "Модель сформувала помірно впевнений результат класифікації. Рекомендована уважна ручна перевірка.",
        )

    return (
        "error",
        "Низька впевненість",
        "Модель не має достатньо впевненого результату. Такий випадок потрібно обов’язково перевірити вручну.",
    )


def collect_sample_files() -> list[Path]:
    sample_files: list[Path] = []

    if not SAMPLE_DIR.exists():
        return sample_files

    for suffix in ALLOWED_SUFFIXES:
        sample_files.extend(sorted(SAMPLE_DIR.glob(f"*{suffix}")))
        sample_files.extend(sorted(SAMPLE_DIR.glob(f"*{suffix.upper()}")))

    return sorted(set(sample_files), key=lambda p: p.name.lower())


def safe_report_stem(filename_hint: Optional[str]) -> str:
    raw_name = Path(filename_hint or "uploaded_image").stem
    safe_name = re.sub(r"[^A-Za-zА-Яа-яІіЇїЄєҐґ0-9_-]+", "_", raw_name).strip("_")
    return safe_name or "uploaded_image"


def image_to_base64_png(image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def load_model_comparison_rows() -> list[dict]:
    if not MODEL_COMPARISON_PATH.exists():
        return [
            {
                "Архітектура": "ResNet18 baseline",
                "Тип": "CNN baseline",
                "Статус": "Оцінено",
                "Accuracy": TEST_ACCURACY,
                "Precision": "див. classification report",
                "Recall": "див. classification report",
                "F1-score": "див. classification report",
                "Примітка": f"{TEST_CORRECT}/{TEST_TOTAL} правильних класифікацій",
            },
            {
                "Архітектура": "SE-ResNet18 Attention",
                "Тип": "CNN + Squeeze-and-Excitation Attention",
                "Статус": "Підключено у коді",
                "Accuracy": "оцінюється тим самим evaluate-скриптом",
                "Precision": "після запуску scripts/evaluate_models.py",
                "Recall": "після запуску scripts/evaluate_models.py",
                "F1-score": "після запуску scripts/evaluate_models.py",
                "Примітка": "Attention-based архітектура для фінальної відповідності ТЗ",
            },
        ]

    try:
        payload = json.loads(MODEL_COMPARISON_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

    rows = payload.get("models", payload if isinstance(payload, list) else [])
    normalized_rows: list[dict] = []

    for item in rows:
        normalized_rows.append(
            {
                "Архітектура": item.get("architecture", item.get("model", "—")),
                "Тип": item.get("family", item.get("type", "—")),
                "Статус": item.get("status", "—"),
                "Accuracy": item.get("accuracy", "—"),
                "Precision": item.get("precision_macro", item.get("precision", "—")),
                "Recall": item.get("recall_macro", item.get("recall", "—")),
                "F1-score": item.get("f1_macro", item.get("f1", "—")),
                "Примітка": item.get("note", ""),
            }
        )

    return normalized_rows


def render_status_card(title: str, text: str, variant: str = "info") -> None:
    safe_variant = variant if variant in {"info", "success", "warning", "danger"} else "info"
    st.markdown(
        f"""
        <div class="cyto-status-card {safe_variant}">
            <div class="status-title">{html.escape(title)}</div>
            <p class="status-text">{html.escape(text)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detail_card(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="cyto-detail-card">
            <div class="detail-title">{html.escape(title)}</div>
            <p class="detail-text">{html.escape(text)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_meta_card(text: str) -> None:
    st.markdown(
        f"""
        <div class="cyto-meta-card">{html.escape(text)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_model_info() -> None:
    with st.expander("Інформація про модель та відповідність ТЗ", expanded=False):
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Архітектура", MODEL_NAME)
        c2.metric("Датасет", DATASET_NAME)
        c3.metric("XAI-метод", XAI_METHOD)
        c4.metric("Тестова точність", TEST_ACCURACY)

        st.write(
            "Фінальна версія орієнтована на вимоги БКР: класифікатор цитологічних зображень, "
            "attention-based CNN архітектура, Grad-CAM пояснення, клінічно-орієнтований висновок "
            "та робоче місце лаборанта/лікаря."
        )

        render_status_card("Attention-механізм", ATTENTION_ADAPTER_NOTE_UA, "info")

        st.write("**Тип моделі:**")
        st.write(MODEL_FAMILY)

        st.write("**Класи моделі:**")
        st.write(", ".join(CLASS_NAMES))

        st.write("**Матриця відповідності ТЗ:**")
        compliance_df = pd.DataFrame(BKR_COMPLIANCE_ITEMS)
        compliance_df = compliance_df.rename(
            columns={
                "requirement": "Вимога ТЗ",
                "implementation": "Реалізація в коді",
                "status": "Статус",
            }
        )
        st.dataframe(compliance_df, use_container_width=True, hide_index=True)

        st.write("**Порівняння моделей / експериментальна частина:**")
        comparison_rows = load_model_comparison_rows()
        if comparison_rows:
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
        else:
            render_status_card(
                "Таблицю порівняння не знайдено",
                "Запусти scripts/evaluate_models.py після навчання або оцінювання моделей.",
                "warning",
            )


def render_match_status(predicted_class: str, expected_class: Optional[str]) -> None:
    match = classes_match(predicted_class, expected_class)

    if expected_class is None:
        render_status_card(
            "Очікуваний клас",
            "Очікуваний клас не визначено з назви файлу.",
            "info",
        )
        return

    render_detail_card("Очікуваний клас", expected_class)

    if match is True:
        render_status_card(
            "Статус перевірки",
            "Класифікація збігається з очікуваним класом.",
            "success",
        )
    else:
        render_status_card(
            "Статус перевірки",
            "Класифікація не збігається з очікуваним класом. Рекомендована ручна перевірка.",
            "warning",
        )


def render_confidence_status(confidence: float) -> None:
    status, title, text = get_confidence_status(confidence)

    if status == "success":
        render_status_card(title, text, "success")
    elif status == "warning":
        render_status_card(title, text, "warning")
    else:
        render_status_card(title, text, "danger")


def render_section_header(title: str, subtitle: Optional[str] = None) -> None:
    subtitle_html = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="section-header">
            <h2>{html.escape(title)}</h2>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_block(
    title: str,
    value: str,
    text: Optional[str] = None,
    variant: str = "info",
) -> None:
    safe_variant = variant if variant in {"success", "warning", "danger", "info"} else "info"
    text_html = f"<p class='clinical-card-text'>{html.escape(text)}</p>" if text else ""
    st.markdown(
        f"""
        <div class="clinical-card {safe_variant}">
            <div class="clinical-card-title">{html.escape(title)}</div>
            <div class="clinical-card-value">{html.escape(value)}</div>
            {text_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_report_payload(
    *,
    filename_hint: Optional[str],
    expected_class: Optional[str],
    result: dict,
) -> dict:
    classification_matches_expected = classes_match(
        result["predicted_class"],
        expected_class,
    )

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "filename": filename_hint,
        "model": {
            "architecture": result.get("model_metadata", {}).get("effective_model_display_name", MODEL_NAME),
            "model_family": result.get("model_metadata", {}).get("model_family", MODEL_FAMILY),
            "checkpoint_model_name": result.get("model_metadata", {}).get("checkpoint_model_name"),
            "effective_model_name": result.get("model_metadata", {}).get("effective_model_name"),
            "has_attention": result.get("model_metadata", {}).get("has_attention"),
            "attention_adapter_enabled": result.get("model_metadata", {}).get("attention_adapter_enabled"),
            "dataset": DATASET_NAME,
            "xai_method": XAI_METHOD,
            "test_accuracy": TEST_ACCURACY,
            "checkpoint_path": str(MODEL_PATH),
        },
        "expected_class": expected_class,
        "predicted_class": result["predicted_class"],
        "classification_matches_expected": classification_matches_expected,
        "prediction_matches_expected": classification_matches_expected,
        "risk_group": result["risk_group"],
        "clinical_conclusion": result["clinical_conclusion"],
        "binary_clinical_label": result.get("binary_clinical_label"),
        "risk_level": result["risk_level"],
        "recommended_action": result["recommended_action"],
        "class_description": result["class_description"],
        "confidence": result["confidence"],
        "confidence_percent": round(result["confidence"] * 100, 2),
        "top3": result["top3"],
        "decision_summary": result["decision_summary"],
        "gradcam_explanation": result["gradcam_explanation"],
        "attention_explanation": result.get("attention_explanation", ATTENTION_EXPLANATION_UA),
        "model_metadata": result.get("model_metadata", {}),
        "bkr_compliance": BKR_COMPLIANCE_ITEMS,
        "system_limitation": result["system_limitation"],
        "note": (
            "This report was generated by an educational prototype. "
            "The system is not a medical diagnostic device and does not replace specialist verification."
        ),
    }


def build_html_report(report_payload: dict, result: dict) -> bytes:
    top3_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['class'])}</td>"
        f"<td>{item['probability'] * 100:.2f}%</td>"
        "</tr>"
        for item in report_payload["top3"]
    )

    original_b64 = image_to_base64_png(result["image"])
    heatmap_b64 = image_to_base64_png(result["heatmap_image"])
    overlay_b64 = image_to_base64_png(result["overlay"])

    html_text = f"""
<!doctype html>
<html lang="uk">
<head>
  <meta charset="utf-8">
  <title>CytoXAI Workstation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #111827; }}
    h1, h2 {{ color: #0f172a; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin: 14px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
    .image-grid img {{ width: 100%; border-radius: 10px; border: 1px solid #e5e7eb; }}
    .label {{ font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; }}
    .value {{ font-size: 18px; font-weight: 700; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    th {{ background: #f8fafc; }}
    .warning {{ color: #92400e; background: #fffbeb; border-color: #fde68a; }}
  </style>
</head>
<body>
  <h1>CytoXAI Workstation — звіт аналізу</h1>
  <p><strong>Дата створення:</strong> {html.escape(str(report_payload['created_at']))}</p>
  <p><strong>Файл:</strong> {html.escape(str(report_payload.get('filename') or 'uploaded_image'))}</p>
  <p><strong>Архітектура:</strong> {html.escape(str(report_payload.get('model', {}).get('architecture', MODEL_NAME)))}</p>
  <p><strong>Тип моделі:</strong> {html.escape(str(report_payload.get('model', {}).get('model_family', MODEL_FAMILY)))}</p>

  <div class="grid">
    <div class="card">
      <div class="label">Клас моделі</div>
      <div class="value">{html.escape(report_payload['predicted_class'])}</div>
    </div>
    <div class="card">
      <div class="label">Впевненість</div>
      <div class="value">{report_payload['confidence_percent']:.2f}%</div>
    </div>
    <div class="card">
      <div class="label">Рівень ризику</div>
      <div class="value">{html.escape(report_payload['risk_level'])}</div>
    </div>
  </div>

  <div class="card">
    <h2>Висновок системи</h2>
    <p>{html.escape(report_payload['clinical_conclusion'])}</p>
    <p>{html.escape(report_payload['decision_summary'])}</p>
  </div>

  <div class="card">
    <h2>Рекомендована дія</h2>
    <p>{html.escape(report_payload['recommended_action'])}</p>
  </div>

  <div class="card">
    <h2>Top-3 класи</h2>
    <table>
      <thead><tr><th>Клас</th><th>Ймовірність</th></tr></thead>
      <tbody>{top3_rows}</tbody>
    </table>
  </div>

  <h2>Візуальне пояснення</h2>
  <div class="image-grid">
    <div><p><strong>Оригінал</strong></p><img src="data:image/png;base64,{original_b64}" alt="Original image"></div>
    <div><p><strong>Grad-CAM</strong></p><img src="data:image/png;base64,{heatmap_b64}" alt="Grad-CAM heatmap"></div>
    <div><p><strong>Overlay</strong></p><img src="data:image/png;base64,{overlay_b64}" alt="Grad-CAM overlay"></div>
  </div>

  <div class="card">
    <h2>Пояснення attention-механізму</h2>
    <p>{html.escape(report_payload.get('attention_explanation', ATTENTION_EXPLANATION_UA))}</p>
  </div>

  <div class="card">
    <h2>Пояснення Grad-CAM</h2>
    <p>{html.escape(report_payload['gradcam_explanation'])}</p>
  </div>

  <div class="card warning">
    <h2>Обмеження системи</h2>
    <p>{html.escape(report_payload['system_limitation'])}</p>
  </div>
</body>
</html>
"""

    return html_text.encode("utf-8")


def save_json_report(report_payload: dict, filename_hint: Optional[str]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = safe_report_stem(filename_hint)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"case_report_{safe_name}_{timestamp}.json"

    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return report_path


def save_html_report(html_report: bytes, filename_hint: Optional[str]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = safe_report_stem(filename_hint)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"case_report_{safe_name}_{timestamp}.html"
    report_path.write_bytes(html_report)
    return report_path


st.markdown(
    """
    <div class="cyto-hero">
        <div class="cyto-hero-kicker">Explainable AI • Cytology Screening Prototype</div>
        <div class="cyto-hero-title">CytoXAI Workstation</div>
        <p class="cyto-hero-text">
            Робоче місце для класифікації цитологічних зображень, оцінки рівня впевненості моделі
            та візуального пояснення рішення за допомогою Grad-CAM.
        </p>
        <div class="cyto-chip-row">
            <span class="cyto-chip">SE-ResNet18 Attention</span>
            <span class="cyto-chip">SIPaKMeD</span>
            <span class="cyto-chip">Grad-CAM heatmap</span>
            <span class="cyto-chip">JSON / HTML звіт</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Вхідні дані")

    sample_files = collect_sample_files()
    selected_sample: Optional[Path] = None

    if sample_files:
        selected_name = st.selectbox(
            "Тестове зображення SIPaKMeD",
            [p.name for p in sample_files],
        )
        selected_sample = SAMPLE_DIR / selected_name
    else:
        render_status_card(
            "Тестові зображення не знайдено",
            "У data/sample_images поки немає підтримуваних зображень.",
            "info",
        )

    uploaded_file = st.file_uploader(
        "Або завантажити власне зображення",
        type=UPLOAD_TYPES,
        help="Підтримуються PNG, JPG, JPEG, BMP, TIF, TIFF.",
    )

    st.divider()
    run_button = st.button("Аналізувати", type="primary", use_container_width=True)

st.markdown(
    f"""
    <div class="limitation-card">
        <div class="block-title">Важливе обмеження</div>
        <div>{html.escape(SYSTEM_LIMITATION_UA)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if MODEL_PATH.exists():
    render_status_card(
        "Checkpoint моделі знайдено",
        f"Використовується збережений checkpoint моделі: {MODEL_PATH}",
        "success",
    )
else:
    render_status_card(
        "Checkpoint моделі не знайдено",
        "Поклади навчений файл у models/cyto_classifier.pth, після чого перезапусти застосунок.",
        "danger",
    )
    st.stop()

render_model_info()

image_source = None
filename_hint: Optional[str] = None
preview_source = None

if uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()
    image_source = BytesIO(uploaded_bytes)
    preview_source = BytesIO(uploaded_bytes)
    filename_hint = uploaded_file.name
elif selected_sample is not None:
    image_source = selected_sample
    preview_source = selected_sample
    filename_hint = selected_sample.name

if image_source is None:
    render_status_card(
        "Немає зображення для аналізу",
        "Додайте зображення у data/sample_images або завантажте файл через sidebar.",
        "info",
    )
    st.stop()

if not run_button:
    render_section_header("Попередній перегляд", "Обране зображення готове до запуску аналізу.")
    st.image(preview_source, caption=filename_hint or "Обране зображення", width=420)
    render_status_card(
        "Готово до аналізу",
        "Натисніть «Аналізувати», щоб виконати класифікацію та побудувати Grad-CAM пояснення.",
        "info",
    )
    st.stop()

try:
    result = predict_image(
        image_source,
        filename_hint=filename_hint,
    )
except Exception as error:
    st.error("Не вдалося виконати аналіз зображення.")
    st.exception(error)
    st.stop()

expected_class = extract_expected_class(filename_hint)

model_metadata = result.get("model_metadata", {})
if model_metadata.get("attention_adapter_enabled"):
    render_status_card(
        "Attention-адаптер активний",
        model_metadata.get("attention_adapter_note", ATTENTION_ADAPTER_NOTE_UA),
        "info",
    )

left, right = st.columns([1.15, 1.0], gap="large")

with left:
    render_section_header("Мікрофотографія", "Вхідне цитологічне зображення, яке передається до моделі класифікації.")
    st.image(
        result["image"],
        caption="Оригінальне зображення",
        use_container_width=True,
    )

with right:
    render_section_header("Результат класифікації", "Передбачений морфологічний клас, confidence та top-3 найбільш імовірні класи.")

    m1, m2 = st.columns([1.35, 0.85], gap="medium")
    m1.metric("Клас", result["predicted_class"])
    m2.metric("Впевненість", f"{result['confidence'] * 100:.2f}%")

    render_match_status(
        predicted_class=result["predicted_class"],
        expected_class=expected_class,
    )

    render_confidence_status(result["confidence"])

    render_detail_card("Опис класу", result["class_description"])

    top3_df = pd.DataFrame(
        [
            {
                "Клас": item["class"],
                "Ймовірність, %": round(item["probability"] * 100, 2),
            }
            for item in result["top3"]
        ]
    )

    st.dataframe(
        top3_df,
        use_container_width=True,
        hide_index=True,
    )

    render_meta_card(
        f"Активна архітектура: {model_metadata.get('effective_model_display_name', MODEL_NAME)} · "
        f"Device: {model_metadata.get('device', 'auto')}"
    )

render_section_header(
    "Клінічно-орієнтована інтерпретація",
    "Текстовий висновок системи подано як допоміжну інтерпретацію для лаборанта або лікаря, а не як остаточний діагноз.",
)

summary_col, risk_col, action_col = st.columns(3)

with summary_col:
    conclusion_variant = "success" if "Норма" in result["clinical_conclusion"] else "warning"
    if "Підозрілі" in result["clinical_conclusion"]:
        conclusion_variant = "danger"
    render_info_block(
        "Висновок системи",
        result["clinical_conclusion"],
        result["decision_summary"],
        variant=conclusion_variant,
    )

with risk_col:
    risk_variant = "success" if "низ" in result["risk_level"].lower() or "норм" in result["risk_level"].lower() else "warning"
    if "вис" in result["risk_level"].lower() or "підоз" in result["risk_level"].lower():
        risk_variant = "danger"
    render_info_block(
        "Рівень ризику",
        result["risk_level"],
        f"Ризик-група: {result['risk_group']}",
        variant=risk_variant,
    )

with action_col:
    render_info_block(
        "Рекомендована дія",
        "Ручна верифікація",
        result["recommended_action"],
        variant="info",
    )

st.markdown(
    f"""
    <div class="xai-note">
        <div class="block-title">Attention-механізм моделі</div>
        <div class="block-text">{html.escape(ATTENTION_EXPLANATION_UA)}</div>
        <div class="small-muted" style="margin-top: 8px;">
            У фінальній архітектурі SE-блоки виконують каналову увагу, а Grad-CAM окремо показує
            просторові області зображення, які найбільше вплинули на класифікацію.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_section_header(
    "Візуальне пояснення рішення",
    "Grad-CAM підсвічує області зображення, які найбільше вплинули на результат класифікації.",
)

c1, c2 = st.columns(2)

with c1:
    st.image(
        result["heatmap_image"],
        caption="Grad-CAM heatmap",
        use_container_width=True,
    )

with c2:
    st.image(
        result["overlay"],
        caption="Overlay: heatmap поверх зображення",
        use_container_width=True,
    )

st.markdown(
    f"""
    <div class="xai-note">
        <div class="block-title">Пояснення Grad-CAM</div>
        <div class="block-text">{html.escape(GRADCAM_EXPLANATION_UA)}</div>
        <div class="small-muted" style="margin-top: 8px;">
            Теплова карта не є самостійним діагнозом. Вона показує, на які ділянки зображення
            модель звернула найбільшу увагу під час класифікації.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="limitation-card">
        <div class="block-title">Обмеження системи</div>
        <div>{html.escape(SYSTEM_LIMITATION_UA)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


report_payload = build_report_payload(
    filename_hint=filename_hint,
    expected_class=expected_class,
    result=result,
)

html_report = build_html_report(report_payload, result)
json_report_path = save_json_report(
    report_payload=report_payload,
    filename_hint=filename_hint,
)
html_report_path = save_html_report(
    html_report=html_report,
    filename_hint=filename_hint,
)

render_meta_card(f"JSON-звіт збережено: {json_report_path}")
render_meta_card(f"HTML-звіт збережено: {html_report_path}")

json_col, html_col = st.columns(2)

with json_col:
    st.download_button(
        label="Завантажити JSON-звіт",
        data=json.dumps(report_payload, ensure_ascii=False, indent=2),
        file_name=f"case_report_{safe_report_stem(filename_hint)}.json",
        mime="application/json",
        use_container_width=True,
    )

with html_col:
    st.download_button(
        label="Завантажити HTML-звіт",
        data=html_report,
        file_name=f"case_report_{safe_report_stem(filename_hint)}.html",
        mime="text/html",
        use_container_width=True,
    )
