"""FastAPI backend: image classification with an open-source Vision Transformer."""
import io
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Any Hugging Face image-classification model works here (override via env var).
MODEL_NAME = os.getenv("MODEL_NAME", "google/vit-base-patch16-224")
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_UPLOAD_MB = 10

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state["processor"] = AutoImageProcessor.from_pretrained(MODEL_NAME)
    state["model"] = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(device).eval()
    state["device"] = device
    yield
    state.clear()


app = FastAPI(title="ViT Image Classifier", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": state.get("device")}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image larger than {MAX_UPLOAD_MB} MB.")

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not read the image.")

    inputs = state["processor"](images=image, return_tensors="pt").to(state["device"])
    with torch.no_grad():
        logits = state["model"](**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]

    top = torch.topk(probs, k=min(TOP_K, probs.numel()))
    id2label = state["model"].config.id2label
    predictions = [
        {"label": id2label[i.item()], "confidence": round(p.item(), 4)}
        for p, i in zip(top.values, top.indices)
    ]
    return {"model": MODEL_NAME, "predictions": predictions}
