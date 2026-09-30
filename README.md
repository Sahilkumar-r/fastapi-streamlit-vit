# ViT Image Classifier

A minimal image-identification app: a **FastAPI** backend serving an open-source **Vision Transformer (ViT)** and a **Streamlit** frontend where users upload a picture and get back what is in it (top-5 ImageNet classes with confidence). Classification only, no chat.

## Architecture

```
Streamlit (frontend/app.py)  --POST /predict-->  FastAPI (backend/main.py)  -->  ViT (Hugging Face)
```

## Model

Default: [`google/vit-base-patch16-224`](https://huggingface.co/google/vit-base-patch16-224), free and open-source (Apache-2.0), pre-trained on ImageNet-21k and fine-tuned on ImageNet-1k (1,000 classes).

Swap in any other Hugging Face image-classification model via an environment variable:

```bash
MODEL_NAME=google/vit-large-patch16-224 uvicorn backend.main:app
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Terminal 1 (backend; the model downloads on first start):

```bash
uvicorn backend.main:app --reload --port 8000
```

Terminal 2 (frontend):

```bash
streamlit run frontend/app.py
```

Open http://localhost:8501, upload an image, click **Classify**.

API docs: http://localhost:8000/docs

## API

| Method | Endpoint   | Description                          |
|--------|------------|--------------------------------------|
| GET    | `/health`  | Status, model name, device           |
| POST   | `/predict` | Multipart `file` -> top-K predictions |

Example:

```bash
curl -X POST -F "file=@cat.jpg" http://localhost:8000/predict
```

## Configuration

| Variable     | Default                        | Used by  |
|--------------|--------------------------------|----------|
| `MODEL_NAME` | `google/vit-base-patch16-224`  | backend  |
| `TOP_K`      | `5`                            | backend  |
| `API_URL`    | `http://localhost:8000`        | frontend |

## Project structure

```
vit-image-classifier/
├── backend/main.py
├── frontend/app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Limitations

- Predictions are limited to the 1,000 ImageNet classes.
- CPU inference works but is slower than GPU.
