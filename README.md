# Torneo Fact-Checking — Vietnamese NLP

End-to-end pipeline for Vietnamese fact-checking using [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base).

| Component | File |
|-----------|------|
| Training  | `train.py` |
| Inference API | `app.py` |
| Docker container | `Dockerfile` |
| Azure deployment | `azure/deploy.sh` |

---

## 1 · Dataset

The dataset is hosted on Kaggle: [haisemei/fact-checking-dataset-label](https://www.kaggle.com/datasets/haisemei/fact-checking-dataset-label).

To download it programmatically:

```python
import kagglehub
path = kagglehub.dataset_download("haisemei/fact-checking-dataset-label")
print("Path to dataset files:", path)
```

> **Note:** You need a `~/.kaggle/kaggle.json` credentials file (or the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables).

---

## 2 · Training locally

### Prerequisites

```bash
pip install -r requirements.txt -r requirements-train.txt
```

### Run

```bash
python train.py \
  --epochs 5 \
  --batch_size 16 \
  --lr 2e-5 \
  --output_dir ./model
```

The script:
1. Downloads the dataset automatically via `kagglehub`.
2. Detects the claim/evidence/label columns.
3. Fine-tunes `xlm-roberta-base` for binary classification (`SUPPORTED` / `REFUTED`).
4. Saves the best checkpoint (by validation accuracy) to `--output_dir`.

---

## 3 · Running the inference API locally

```bash
pip install -r requirements.txt
MODEL_DIR=./model uvicorn app:app --host 0.0.0.0 --port 8000
```

### Request format

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"claim": "Hà Nội là thủ đô của Việt Nam.", "evidence": "Hà Nội là thủ đô và là thành phố đứng đầu về diện tích tự nhiên của Việt Nam."}'
```

### Response format

```json
{
  "predicted_label": "SUPPORTED"
}
```

The `predicted_label` field is always either `"SUPPORTED"` or `"REFUTED"`.

---

## 4 · Docker

### Build the image (model must be trained first)

```bash
docker build -t fact-checking-api:latest .
```

### Run with Docker Compose

```bash
docker compose up
```

The API will be available at `http://localhost:8000`.

---

## 5 · Azure deployment

### Prerequisites

- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) installed
- Logged in: `az login`
- Docker installed

### Deploy

```bash
chmod +x azure/deploy.sh
./azure/deploy.sh
```

The script:
1. Creates an Azure Resource Group.
2. Creates an Azure Container Registry (ACR).
3. Builds & pushes the Docker image to ACR using `az acr build`.
4. Creates an Azure Container Apps environment.
5. Deploys the container and prints the public HTTPS endpoint.

Override defaults with environment variables:

```bash
RESOURCE_GROUP=my-rg \
LOCATION=westeurope \
ACR_NAME=myfactcheckingacr \
./azure/deploy.sh
```

---

## 6 · API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/predict` | Predict label for a claim |

### POST /predict

**Request body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `claim` | string | ✅ | The statement to fact-check |
| `evidence` | string | ❌ | Supporting context (improves accuracy) |

**Response body**

```json
{
  "predicted_label": "SUPPORTED" | "REFUTED"
}
```

---

## 7 · Tests

```bash
pip install -r requirements.txt pytest httpx
python -m pytest tests/ -v
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `./model` | Path to the saved model directory |
| `MAX_LENGTH` | `256` | Tokeniser maximum sequence length |
| `DEVICE` | auto | `cuda` or `cpu` |
