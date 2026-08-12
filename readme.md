# MNIST Handwritten Digit Classifier

CNN-модель на PyTorch для распознавания рукописных цифр (0–9). Обучена на датасете MNIST, обёрнута в REST API (FastAPI) с альтернативным Streamlit-интерфейсом для локальной демонстрации.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.1x-teal)]()
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.32%25-brightgreen)]()

---

## Результаты

Метрики из обучения (`MNIST_Numbers.ipynb`, 15 эпох):

| Выборка | Accuracy |
|---------|----------|
| Train   | 99.34%   |
| Test    | 98.32%   |

- Функция потерь: `CrossEntropyLoss`
- Оптимизатор: `Adam`, `lr=0.001`
- Обучение: 15 эпох, `batch_size=32`, GPU (T4, Google Colab)

---

## Архитектура модели
```
Conv2d(1 → 16, kernel=3, padding=1) → ReLU → MaxPool2d(2)
Flatten → Linear(16×14×14 → 64) → ReLU → Dropout(0.25) → Linear(64 → 10)
```
Компактная CNN — один свёрточный блок вместо глубокой сети, чего достаточно для MNIST (28×28, один канал).

---

## Датасет

- **Источник:** MNIST (Yann LeCun / NYU) — [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist)
- **Размер:** 70,000 изображений (60k train / 10k test)
- **Формат:** 28×28, grayscale, 10 классов (цифры 0–9)
- Загружается напрямую через `torchvision.datasets.MNIST`, ручной подготовки не требует

---

## Инференс на реальных изображениях

Модель обучена на чистом MNIST, но `main.py` содержит отдельный пайплайн предобработки (`preprocess()`) под фото/скан произвольного размера и формата:

1. Конвертация в grayscale + инверсия (MNIST — белым по чёрному)
2. Обрезка по bounding box цифры (убирает пустые поля)
3. Паддинг до квадрата + отступ 20% (как в оригинальном датасете)
4. Resize до 28×28 (LANCZOS)

Это отдельный код, не пересекающийся с обучающим ноутбуком — нужен, потому что реальное фото с телефона или canvas-рисунок не соответствуют формату MNIST "из коробки".

---

## Как запустить

Проект поддерживает два режима — активен FastAPI, Streamlit-версия закомментирована в этом же файле.

### FastAPI (текущий режим)

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Документация API: `http://127.0.0.1:8000/docs`

**Пример запроса:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@digit.png"
```

**Ответ:**
```json
{
  "digit": 7,
  "confidence": 98.4,
  "all_probabilities": {"0": 0.1, "1": 0.0, ..., "7": 98.4, ...}
}
```

### Streamlit (альтернатива)

Раскомментировать Streamlit-блок в `main.py` и закомментировать FastAPI-блок, затем:

```bash
streamlit run main.py
```

Интерфейс — canvas для рисования цифры мышью (`streamlit-drawable-canvas`) + вывод предсказания с confidence и bar-chart вероятностей по всем 10 классам.

---

## Стек

| Категория     | Инструменты                                  |
|---------------|-----------------------------------------------|
| Язык          | Python 3.10+                                   |
| ML            | PyTorch, torchvision                           |
| API           | FastAPI, Uvicorn                               |
| UI (опция)    | Streamlit, streamlit-drawable-canvas           |
| Обработка изображений | Pillow, NumPy                          |
| Обучение      | Google Colab (GPU T4)                          |

---

## Структура проекта
```
ml_MNIST_Numbers/
├── .gitignore
├── readme.md
├── requirements.txt
└── MNIST_Numbers/
    ├── MNIST_Numbers.ipynb
    ├── model_CheckImage_MNIST_Numbers.pth
    ├── main.py
    ├── numbers test/
    │   ├── 0.png
    │   ├── 1.png
    │   ├── 2.png
    │   ├── 3.png
    │   ├── 4.png
    │   ├── 5.png
    │   ├── 6.png
    │   └── 7.png
    ├── test.png
    └── test2.png
```

---