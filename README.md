# CytoXAI Workstation

Програмний прототип для бакалаврської кваліфікаційної роботи за темою: **«Розробка модуля інтерпретації рішень нейронних мереж для цитологічного скринінгу»**.

Система реалізує робоче місце лаборанта/лікаря для навчального сценарію цитологічного скринінгу: класифікує цитологічне зображення, показує рівень впевненості моделі, формує Grad-CAM heatmap, overlay-візуалізацію, клінічно-орієнтований висновок, рекомендовану дію та звіт у JSON/HTML.

## Фінальна відповідність ТЗ

У фінальному патчі додано формальне закриття всіх ключових вимог ТЗ:

1. **Класифікатор цитологічних зображень** — SE-ResNet18 Attention / ResNet18 backbone, 5 класів SIPaKMeD.
2. **CNN з механізмом Attention** — реалізовано Squeeze-and-Excitation channel attention у `src/model.py`.
3. **XAI-модуль** — Grad-CAM heatmap та overlay поверх мікрофотографії.
4. **Клінічно-орієнтована інтерпретація** — норма / підозрілі морфологічні зміни / пограничний випадок.
5. **Робоче місце лаборанта/лікаря** — Streamlit UI з мікрофотографією, confidence, top-3, risk-level, рекомендованою дією та звітами.
6. **Технічний стек** — Python, PyTorch, OpenCV-compatible dependency, Streamlit.
7. **Експериментальна частина** — додано таблицю порівняння моделей і скрипти для fine-tuning/evaluation.

## Основна функціональність

1. вибір тестового зображення або завантаження власного файлу;
2. preprocessing зображення;
3. 5-класова класифікація цитологічного зображення;
4. active architecture: `SE-ResNet18 Attention`;
5. відображення confidence та top-3 класів;
6. формування спрощеної групи ризику;
7. формування висновку системи: норма, підозрілі морфологічні зміни або пограничний випадок;
8. побудова Grad-CAM heatmap;
9. побудова overlay heatmap поверх оригінального зображення;
10. відображення рекомендованої дії для ручної верифікації;
11. експорт JSON-звіту та HTML-звіту;
12. вбудована матриця відповідності ТЗ.

## Структура проєкту

```text
cytoxai_workstation/
  app.py
  requirements.txt
  data/
    sample_images/
    processed/
      sipakmed_5class/
  models/
    cyto_classifier.pth
    cyto_classifier_attention.pth      # optional after fine-tuning
  src/
    config.py
    model.py
    preprocessing.py
    predict.py
    gradcam.py
  scripts/
    train_attention_model.py
    evaluate_models.py
  outputs/
    evaluation/
      model_comparison.json
    reports/
```

## Запуск застосунку

```powershell
cd "D:\University\NULP\8 сем 4 курс\practic\cytoxai_workstation"
.\.venv\Scripts\activate.bat
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Checkpoint моделі

Застосунок очікує навчений checkpoint:

```text
models/cyto_classifier.pth
```

Якщо файл відсутній, застосунок зупиняється і показує повідомлення про помилку. Демонстраційний режим у фінальній версії не використовується: для аналізу має застосовуватися реальний навчений checkpoint моделі.

## Attention adapter

Фінальна архітектура в коді — `SE-ResNet18 Attention`. Якщо наявний лише checkpoint від ResNet18, система завантажує його як backbone і додає SE attention-блоки з identity-сумісною ініціалізацією. Це не ламає вже отримані результати моделі, але формально переводить архітектуру в CNN із механізмом уваги та дає основу для подальшого fine-tuning.

Для повного fine-tuning attention-моделі:

```powershell
.\.venv\Scripts\python.exe scripts\train_attention_model.py `
  --data-dir data\processed\sipakmed_5class `
  --init-checkpoint models\cyto_classifier.pth `
  --output models\cyto_classifier_attention.pth `
  --epochs 8
```

Після fine-tuning можна або замінити `models/cyto_classifier.pth`, або вручну адаптувати `MODEL_PATH` у `src/config.py`.

## Оцінювання та порівняння моделей

Для генерації актуальної таблиці accuracy / precision / recall / F1:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_models.py `
  --data-dir data\processed\sipakmed_5class\test `
  --checkpoint models\cyto_classifier.pth `
  --model se_resnet18 `
  --output outputs\evaluation\model_comparison.json
```

Після цього таблиця підтягнеться в Streamlit-блоці **«Інформація про модель та відповідність ТЗ»**.

## Дані

Для навчання та тестування використовується датасет SIPaKMeD із п’ятьма морфологічними класами:

- Dyskeratotic;
- Koilocytotic;
- Metaplastic;
- Parabasal;
- Superficial-Intermediate.

## Результати поточної моделі

- Активна архітектура в коді: SE-ResNet18 Attention.
- Backbone / checkpoint: ResNet18.
- Датасет: SIPaKMeD.
- Кількість класів: 5.
- Тестова точність baseline checkpoint: 96.71%.
- XAI-метод: Grad-CAM.

## Важливе обмеження

CytoXAI Workstation є навчальним програмним прототипом. Система не є медичним діагностичним засобом, не встановлює діагноз і не замінює рішення лікаря або лаборанта.
