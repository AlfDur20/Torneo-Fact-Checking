"""
train.py — Fine-tune XLM-RoBERTa on the Vietnamese fact-checking dataset.

Usage:
    python train.py [--epochs 5] [--batch_size 16] [--lr 2e-5] [--output_dir ./model]

The script:
  1. Downloads the dataset from Kaggle (haisemei/fact-checking-dataset-label).
  2. Explores the CSV columns and maps them to claim / evidence / label.
  3. Fine-tunes xlm-roberta-base for binary sequence classification
     (SUPPORTED=1 / REFUTED=0).
  4. Saves the model, tokenizer and a label_map.json to --output_dir.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "xlm-roberta-base"
LABEL_MAP = {"SUPPORTED": 1, "REFUTED": 0}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
MAX_LENGTH = 256  # tokens for claim + evidence combined


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------
class FactCheckDataset(Dataset):
    """Tokenised claim-evidence pairs with binary labels."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def download_dataset() -> str:
    """Download dataset via kagglehub and return the local path."""
    import kagglehub  # imported here so training deps are only required for train.py

    path = kagglehub.dataset_download("haisemei/fact-checking-dataset-label")
    print(f"Dataset downloaded to: {path}")
    return path


def _find_csv(dataset_path: str) -> str:
    """Locate the first CSV file inside the dataset folder."""
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.endswith(".csv"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"No CSV found under {dataset_path}")


def _detect_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str]:
    """
    Heuristically detect claim, evidence and label column names.
    Returns (claim_col, evidence_col, label_col).
    evidence_col may be None if not found.
    """
    cols_lower = {c.lower(): c for c in df.columns}

    # label column
    label_col = None
    for candidate in ("label", "verdict", "class", "gold_label"):
        if candidate in cols_lower:
            label_col = cols_lower[candidate]
            break
    if label_col is None:
        raise ValueError(f"Cannot find label column. Available columns: {list(df.columns)}")

    # claim column
    claim_col = None
    for candidate in ("claim", "statement", "text", "sentence", "question"):
        if candidate in cols_lower:
            claim_col = cols_lower[candidate]
            break

    # evidence column
    evidence_col = None
    for candidate in ("evidence", "context", "passage", "article", "document"):
        if candidate in cols_lower:
            evidence_col = cols_lower[candidate]
            break

    # fallback: first non-label text column
    if claim_col is None:
        for col in df.columns:
            if col != label_col and df[col].dtype == object:
                claim_col = col
                break

    if claim_col is None:
        raise ValueError(f"Cannot find claim/text column. Available: {list(df.columns)}")

    print(f"Detected columns → claim: '{claim_col}', evidence: '{evidence_col}', label: '{label_col}'")
    return claim_col, evidence_col, label_col


def _normalise_label(raw: str) -> str | None:
    """Map raw label strings to SUPPORTED / REFUTED."""
    s = str(raw).strip().upper()
    mapping = {
        "SUPPORTED": "SUPPORTED",
        "SUPPORT": "SUPPORTED",
        "TRUE": "SUPPORTED",
        "CORRECT": "SUPPORTED",
        "1": "SUPPORTED",
        "REFUTED": "REFUTED",
        "REFUTE": "REFUTED",
        "FALSE": "REFUTED",
        "INCORRECT": "REFUTED",
        "0": "REFUTED",
    }
    return mapping.get(s)


def load_dataframe(dataset_path: str) -> pd.DataFrame:
    """Load, inspect and clean the raw dataset CSV."""
    csv_path = _find_csv(dataset_path)
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Raw shape: {df.shape}")
    print(df.head(3).to_string())

    claim_col, evidence_col, label_col = _detect_columns(df)

    # Normalise labels and drop unknowns
    df["_label"] = df[label_col].apply(_normalise_label)
    unknown = df["_label"].isna()
    if unknown.sum():
        unique_unknown = df.loc[unknown, label_col].unique()
        print(f"Warning: dropping {unknown.sum()} rows with unknown labels: {unique_unknown}")
    df = df[~unknown].copy()

    # Build input text: "claim [SEP] evidence" when evidence is present
    if evidence_col and evidence_col in df.columns:
        df["_text"] = df[claim_col].fillna("").astype(str) + " </s></s> " + df[evidence_col].fillna("").astype(str)
    else:
        df["_text"] = df[claim_col].fillna("").astype(str)

    df["_label_id"] = df["_label"].map(LABEL_MAP)
    print(f"Clean shape: {df.shape}")
    print(df["_label"].value_counts().to_string())
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data
    dataset_path = download_dataset()
    df = load_dataframe(dataset_path)

    texts = df["_text"].tolist()
    labels = df["_label_id"].tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # 2. Tokeniser
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = FactCheckDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = FactCheckDataset(X_val, y_val, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # 3. Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
    )
    model.to(device)

    # 4. Optimiser + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps
    )

    # 5. Training loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(batch["labels"].cpu().numpy())

        val_acc = np.mean(np.array(all_preds) == np.array(all_true))
        print(f"Epoch {epoch}/{args.epochs} — loss: {avg_loss:.4f}, val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best checkpoint
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f}) → {args.output_dir}")

    print("\nFinal validation report:")
    print(classification_report(all_true, all_preds, target_names=list(LABEL_MAP.keys())))

    # Save label map alongside the model
    label_map_path = os.path.join(args.output_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL_MAP}, f, indent=2)
    print(f"Label map saved to {label_map_path}")
    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vietnamese fact-checking model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str, default="./model")
    args = parser.parse_args()
    train(args)
