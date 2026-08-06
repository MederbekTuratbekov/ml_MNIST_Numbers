# MNIST_Numbers/main.py

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


BASE_DIR = Path(__file__).parent


# ── Model ──────────────────────────────────────────────────────────────────────
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel().to(device)
    model.load_state_dict(torch.load(BASE_DIR / "model_MNIST.pth", map_location=device))
    model.eval()
    return model, device


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(canvas_image: np.ndarray) -> torch.Tensor:
    img = Image.fromarray(canvas_image.astype("uint8")).convert("L")
    img = ImageOps.invert(img)

    arr = np.array(img)
    coords = np.argwhere(arr > 20)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        img = img.crop((x0, y0, x1 + 1, y1 + 1))

    w, h = img.size
    side = max(w, h)
    padded = Image.new("L", (side, side), color=0)
    padded.paste(img, ((side - w) // 2, (side - h) // 2))
    padded = ImageOps.expand(padded, border=side // 5, fill=0)

    resized = padded.resize((28, 28), Image.LANCZOS)
    tensor = torch.tensor(np.array(resized), dtype=torch.float32) / 255.0
    return tensor.unsqueeze(0).unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT — закомментировано, раскомментируй этот блок и закомментируй FastAPI ниже
# запуск: streamlit run main.py
# ══════════════════════════════════════════════════════════════════════════════

# st.title("Распознавание рукописных цифр")
# st.write("Нарисуй цифру от 0 до 9")
#
# model, device = load_model()
#
# canvas_result = st_canvas(
#     fill_color="black",
#     stroke_width=18,
#     stroke_color="black",
#     background_color="white",
#     width=280,
#     height=280,
#     drawing_mode="freedraw",
#     key="canvas",
# )
#
# if canvas_result.image_data is not None:
#     if st.button("Распознать"):
#         x = preprocess(canvas_result.image_data).to(device)
#
#         with torch.no_grad():
#             logits = model(x)
#             proba  = torch.softmax(logits, dim=1)[0]
#             pred   = int(torch.argmax(proba).item())
#
#         st.subheader(f"Это цифра: {pred}")
#         st.write(f"Уверенность: {proba[pred].item() * 100:.1f}%")
#
#         st.bar_chart({str(i): proba[i].item() for i in range(10)})


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI — активно
# запуск: uvicorn main:app --reload --port 8000
# ══════════════════════════════════════════════════════════════════════════════

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
import uvicorn
import io

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.device = load_model()
    yield


app = FastAPI(title="MNIST Digit Classifier", lifespan=lifespan)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    x = preprocess(np.array(img)).to(app.state.device)

    with torch.no_grad():
        logits = app.state.model(x)
        proba  = torch.softmax(logits, dim=1)[0]
        pred   = int(torch.argmax(proba).item())

    return {
        "digit":       pred,
        "confidence":  round(proba[pred].item() * 100, 2),
        "all_probabilities": {str(i): round(proba[i].item() * 100, 2) for i in range(10)},
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)