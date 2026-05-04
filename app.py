"""
app.py — FastAPI inference server for Vietnamese fact-checking.

Endpoints:
  POST /predict
    Body: {"claim": "...", "evidence": "..."}   (evidence is optional)
    Returns: {"predicted_label": "SUPPORTED" | "REFUTED"}

  GET /health
    Returns: {"status": "ok"}

Environment variables:
  MODEL_DIR   Path to the saved model directory (default: ./model)
  MAX_LENGTH  Tokeniser max length (default: 256)
  DEVICE      "cuda" | "cpu"  (auto-detected if unset)
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (read at startup)
# ---------------------------------------------------------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "./model")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
_DEVICE_ENV = os.getenv("DEVICE", "")

# ---------------------------------------------------------------------------
# Global model state (populated during lifespan startup)
# ---------------------------------------------------------------------------
_state: dict = {}


def _load_model():
    device_str = _DEVICE_ENV if _DEVICE_ENV else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info("Loading model from %s on %s …", MODEL_DIR, device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    # Load id→label mapping saved during training
    label_map_path = os.path.join(MODEL_DIR, "label_map.json")
    if os.path.exists(label_map_path):
        with open(label_map_path) as f:
            meta = json.load(f)
        id2label = {int(k): v for k, v in meta["id2label"].items()}
    else:
        # Fallback: derive from model config
        id2label = {int(k): v for k, v in model.config.id2label.items()}

    logger.info("Model loaded. Labels: %s", id2label)
    return tokenizer, model, device, id2label


# ---------------------------------------------------------------------------
# Lifespan (FastAPI startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    tokenizer, model, device, id2label = _load_model()
    _state["tokenizer"] = tokenizer
    _state["model"] = model
    _state["device"] = device
    _state["id2label"] = id2label
    yield
    _state.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Vietnamese Fact-Checking API",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    claim: str
    evidence: Optional[str] = None


class PredictResponse(BaseModel):
    predicted_label: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.claim or not request.claim.strip():
        raise HTTPException(status_code=422, detail="'claim' must be a non-empty string.")

    # Build input text identical to the format used during training
    if request.evidence:
        text = request.claim.strip() + " </s></s> " + request.evidence.strip()
    else:
        text = request.claim.strip()

    tokenizer = _state["tokenizer"]
    model = _state["model"]
    device = _state["device"]
    id2label = _state["id2label"]

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        logits = model(**encoding).logits

    predicted_id = int(torch.argmax(logits, dim=-1).item())
    label = id2label[predicted_id]

    return PredictResponse(predicted_label=label)


# ---------------------------------------------------------------------------
# Allow running with:  python app.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
