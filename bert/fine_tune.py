"""
Legal-BERT Fine-Tuning Script for Clause Classification

Fine-tunes nlpaueb/legal-bert-base-uncased on the CUAD dataset for
multi-label clause type classification.

Usage:
    python -m bert.fine_tune                 # Full training
    python -m bert.fine_tune --dry-run       # Quick test with 100 samples
    python -m bert.fine_tune --epochs 3      # Override epochs
"""

import argparse
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from bert.config import (
    BASE_MODEL_NAME,
    OUTPUT_DIR,
    LEARNING_RATE,
    NUM_EPOCHS,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    MAX_SEQ_LENGTH,
    GRADIENT_ACCUMULATION_STEPS,
    PREDICTION_THRESHOLD,
    NUM_LABELS,
    ID2LABEL,
    LABEL2ID,
)
from bert.dataset_utils import build_classification_dataset, tokenize_dataset


# ── Device Detection ─────────────────────────────────────────────────────────

def _detect_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon MPS")
    else:
        device = "cpu"
        print("💻 Using CPU (training will be slower)")
    return device


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """
    Compute multi-label classification metrics.
    Uses sigmoid + threshold to convert logits to binary predictions.
    """
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs > PREDICTION_THRESHOLD).astype(int)
    labels = labels.astype(int)

    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
    }


# ── Main Training Loop ───────────────────────────────────────────────────────

def train(
    epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    dry_run: bool = False,
    output_dir: str = OUTPUT_DIR,
):
    """Run the full fine-tuning pipeline."""
    device = _detect_device()

    # ── 1. Prepare Dataset ───────────────────────────────────────────────
    print("\n Preparing dataset...")
    max_samples = 100 if dry_run else None
    dataset = build_classification_dataset(max_samples=max_samples)

    print(f"   Train:      {len(dataset['train'])} examples")
    print(f"   Validation: {len(dataset['validation'])} examples")

    # ── 2. Tokenizer ─────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenized = tokenize_dataset(dataset, tokenizer)

    # ── 3. Model ─────────────────────────────────────────────────────────
    print(f"\n Loading model: {BASE_MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        problem_type="multi_label_classification",
    )

    # ── 4. Training Arguments ────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,

        # Epochs & batching
        num_train_epochs=1 if dry_run else epochs,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        # Optimizer
        learning_rate=lr,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",

        # Evaluation & saving
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,

        # Logging
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        report_to="none",  # Set to "wandb" if you want W&B tracking

        # Performance
        fp16=(device == "cuda"),
        dataloader_num_workers=0,

        # Seed
        seed=42,
    )

    # ── 5. Trainer ───────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── 6. Train! ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🏋️  STARTING FINE-TUNING")
    print("=" * 60)
    print(f"   Model:      {BASE_MODEL_NAME}")
    print(f"   Labels:     {NUM_LABELS} clause types")
    print(f"   Epochs:     {1 if dry_run else epochs}")
    print(f"   Batch size: {TRAIN_BATCH_SIZE} × {GRADIENT_ACCUMULATION_STEPS} (gradient accumulation)")
    print(f"   LR:         {lr}")
    print(f"   Device:     {device}")
    print("=" * 60 + "\n")

    train_result = trainer.train()

    # ── 7. Evaluate ──────────────────────────────────────────────────────
    print("\n📈 Running final evaluation...")
    metrics = trainer.evaluate()
    print("\n🎯 Final Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")

    # ── 8. Save ──────────────────────────────────────────────────────────
    print(f"\n💾 Saving model to '{output_dir}'...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\nFin e-tuning complete!")
    print(f"   Model saved to: {os.path.abspath(output_dir)}")
    print(f"   Total training time: {train_result.metrics.get('train_runtime', 0):.1f}s")

    return metrics


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Legal-BERT for clause classification on CUAD"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick test run with 100 samples and 1 epoch",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for the trained model (default: {OUTPUT_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        epochs=args.epochs,
        lr=args.lr,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
    )
