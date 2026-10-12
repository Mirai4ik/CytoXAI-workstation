from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample_images"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
REPORT_DIR = OUTPUT_DIR / "reports"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"
MODEL_COMPARISON_PATH = EVALUATION_DIR / "model_comparison.json"

IMAGE_SIZE = 224
NUM_CLASSES = 5

# Активна архітектура для фінальної версії БКР.
# Вона формально закриває вимогу ТЗ: CNN з механізмом уваги.
# Якщо є тільки ResNet18 checkpoint, він завантажується як backbone, а SE-блоки
# ініціалізуються identity-сумісно, щоб не ламати вже навчену модель.
ACTIVE_MODEL_NAME = "se_resnet18"
BASELINE_MODEL_NAME = "resnet18"
MODEL_DISPLAY_NAME = "SE-ResNet18 Attention"
MODEL_FAMILY = "CNN з механізмом Squeeze-and-Excitation Attention"
MODEL_BACKBONE_NAME = "ResNet18"
XAI_METHOD = "Grad-CAM"
DATASET_NAME = "SIPaKMeD"
TEST_ACCURACY = "96.71%"
TEST_CORRECT = 588
TEST_TOTAL = 608

ALLOW_RESNET18_CHECKPOINT_ATTENTION_ADAPTER = True
ATTENTION_ADAPTER_NOTE_UA = (
    "Фінальна версія використовує SE-ResNet18 як attention-based CNN. "
    "Якщо доступний лише навчений ResNet18 checkpoint, ваги backbone завантажуються з нього, "
    "а SE-блоки ініціалізуються identity-сумісно. Це дозволяє зберегти стабільність "
    "поточної навченої моделі та водночас мати архітектурний механізм уваги для подальшого fine-tuning."
)

# Raw class names used by the Colab ImageFolder checkpoint.
RAW_CLASS_TO_DISPLAY = {
    "dyskeratotic": "Dyskeratotic",
    "koilocytotic": "Koilocytotic",
    "metaplastic": "Metaplastic",
    "parabasal": "Parabasal",
    "superficial_intermediate": "Superficial-Intermediate",
    "superficial-intermediate": "Superficial-Intermediate",
    "Superficial-Intermediate": "Superficial-Intermediate",
    "Parabasal": "Parabasal",
    "Koilocytotic": "Koilocytotic",
    "Dyskeratotic": "Dyskeratotic",
    "Metaplastic": "Metaplastic",
}

CLASS_NAMES = [
    "Superficial-Intermediate",
    "Parabasal",
    "Koilocytotic",
    "Dyskeratotic",
    "Metaplastic",
]

RISK_GROUPS = {
    "Superficial-Intermediate": "Норма",
    "Parabasal": "Норма",
    "Koilocytotic": "Підозрілі морфологічні зміни",
    "Dyskeratotic": "Підозрілі морфологічні зміни",
    "Metaplastic": "Пограничний випадок / потрібна ручна перевірка",
}

CLINICAL_CONCLUSIONS = {
    "Superficial-Intermediate": "Норма",
    "Parabasal": "Норма",
    "Koilocytotic": "Підозрілі морфологічні зміни",
    "Dyskeratotic": "Підозрілі морфологічні зміни",
    "Metaplastic": "Пограничний випадок / потрібна ручна перевірка",
}

BINARY_CLINICAL_LABELS = {
    "Superficial-Intermediate": "normal",
    "Parabasal": "normal",
    "Koilocytotic": "suspicious",
    "Dyskeratotic": "suspicious",
    "Metaplastic": "borderline_review",
}

RISK_LEVELS = {
    "Superficial-Intermediate": "Низький рівень ризику",
    "Parabasal": "Низький рівень ризику",
    "Koilocytotic": "Підвищений рівень ризику",
    "Dyskeratotic": "Підвищений рівень ризику",
    "Metaplastic": "Невизначений / пограничний рівень ризику",
}

RECOMMENDED_ACTIONS = {
    "Superficial-Intermediate": (
        "Результат можна використовувати як допоміжний орієнтир. "
        "За наявності клінічних підозр або низької якості зображення потрібна ручна перевірка спеціалістом."
    ),
    "Parabasal": (
        "Рекомендована оцінка результату у контексті препарату та клінічної інформації. "
        "За потреби спеціаліст має підтвердити класифікацію вручну."
    ),
    "Koilocytotic": (
        "Рекомендована ручна верифікація спеціалістом, оскільки модель віднесла зображення "
        "до класу з підозрілими морфологічними ознаками."
    ),
    "Dyskeratotic": (
        "Рекомендована пріоритетна ручна верифікація спеціалістом та перегляд ділянок, "
        "підсвічених Grad-CAM."
    ),
    "Metaplastic": (
        "Рекомендована ручна перевірка, оскільки випадок належить до пограничної групи "
        "і потребує оцінки спеціалістом у ширшому морфологічному контексті."
    ),
}

CLASS_DESCRIPTIONS_UA = {
    "Superficial-Intermediate": "Нормальний тип плоских епітеліальних клітин.",
    "Parabasal": "Нормальний або реактивний тип клітин, що потребує оцінки в контексті препарату.",
    "Koilocytotic": "Клітини з морфологічними ознаками, які можуть бути підозрілими для цитологічного скринінгу.",
    "Dyskeratotic": "Клітини з підозрілими морфологічними змінами та підвищеним ризиком патології.",
    "Metaplastic": "Доброякісний або пограничний тип, який бажано перевіряти спеціалістом.",
}

GRADCAM_EXPLANATION_UA = (
    "Grad-CAM формує теплову карту важливості для класу, обраного моделлю. "
    "Жовті, помаранчеві та червоні ділянки на overlay відповідають областям, "
    "які зробили найбільший внесок у класифікацію. У цитологічному сценарії такі "
    "області можуть відповідати зонам ядра, цитоплазми або іншим морфологічним структурам."
)

ATTENTION_EXPLANATION_UA = (
    "Squeeze-and-Excitation Attention оцінює відносну важливість каналів ознак CNN. "
    "Такий механізм дає змогу моделі підсилювати більш інформативні морфологічні ознаки "
    "та послаблювати менш релевантні канали перед фінальною класифікацією."
)

SYSTEM_LIMITATION_UA = (
    "CytoXAI Workstation є навчальним програмним прототипом. Система не є медичним "
    "діагностичним засобом, не встановлює діагноз і не замінює рішення лікаря або лаборанта."
)

BKR_COMPLIANCE_ITEMS = [
    {
        "requirement": "Інтелектуальний класифікатор цитологічних зображень",
        "implementation": "SE-ResNet18 Attention / ResNet18 backbone, 5 класів SIPaKMeD",
        "status": "Виконано",
    },
    {
        "requirement": "Норма / патологія / пограничний висновок",
        "implementation": "Мапінг 5 морфологічних класів у клінічно-орієнтовані групи",
        "status": "Виконано",
    },
    {
        "requirement": "XAI-модуль з тепловою картою",
        "implementation": "Grad-CAM heatmap та overlay поверх мікрофотографії",
        "status": "Виконано",
    },
    {
        "requirement": "Робоче місце лаборанта/лікаря",
        "implementation": "Streamlit Workstation: зображення, клас, confidence, top-3, ризик, рекомендація, звіти",
        "status": "Виконано",
    },
    {
        "requirement": "Python + PyTorch + OpenCV + Streamlit",
        "implementation": "PyTorch inference, Pillow/OpenCV-compatible preprocessing, Streamlit UI, opencv-python у requirements",
        "status": "Виконано",
    },
]

# Must match the Colab training transforms based on ImageNet pretrained ResNet18.
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

MODEL_PATH = MODEL_DIR / "cyto_classifier.pth"
