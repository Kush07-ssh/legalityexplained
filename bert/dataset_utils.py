"""
CUAD Dataset Utilities

Downloads the CUAD (Contract Understanding Atticus Dataset) from Hugging Face
and converts it from SQuAD-style extractive QA format into a multi-label
classification format suitable for fine-tuning Legal-BERT.

CUAD Format (original):
    Each example is a (context, question, answer) triple where the question
    encodes the clause type and the answer is an extracted span (or empty).

Classification Format (target):
    Each unique context paragraph becomes one training example.
    Its label is a multi-hot vector indicating which clause types are present
    (i.e. have non-empty answer spans) in that paragraph.
"""

import re
from collections import defaultdict

import torch
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer

from bert.config import (
    DATASET_NAME,
    BASE_MODEL_NAME,
    MAX_SEQ_LENGTH,
    CLAUSE_LABELS,
    LABEL2ID,
    NUM_LABELS,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_clause_type(question: str) -> str | None:
    """
    CUAD questions follow patterns like:
        'Highlight the parts (if any) of this contract related to
         "Anti-Assignment". ...'
    We extract the clause type from the quoted phrase.
    """
    match = re.search(r'"([^"]+)"', question)
    if match:
        clause = match.group(1).strip()
        # Normalise to match our CLAUSE_LABELS list (title-cased)
        for label in CLAUSE_LABELS:
            if label.lower() == clause.lower():
                return label
    return None


def _has_answer(example: dict) -> bool:
    """Returns True when the example has at least one non-empty answer span."""
    answers = example.get("answers", {})
    texts = answers.get("text", [])
    return any(t.strip() for t in texts)


# ── Main Pipeline ────────────────────────────────────────────────────────────

def build_classification_dataset(
    test_size: float = 0.15,
    seed: int = 42,
    max_samples: int | None = None,
) -> DatasetDict:
    """
    Download CUAD, convert to multi-label classification, and return a
    DatasetDict with 'train' and 'validation' splits.

    Parameters
    ----------
    test_size : float
        Fraction of data reserved for validation.
    seed : int
        Random seed for reproducibility.
    max_samples : int | None
        If set, cap the total dataset size (useful for quick test runs).

    Returns
    -------
    DatasetDict with columns: ['text', 'labels']
        - text:   str — the contract paragraph
        - labels: List[float] — multi-hot vector of length NUM_LABELS
    """
    print(f"Loading CUAD dataset from '{DATASET_NAME}'...")
    raw = load_dataset(
        DATASET_NAME,
        split="train",
        trust_remote_code=True,
        verification_mode="no_checks",
    )

    # ── Step 1: Group by context → set of clause types present ───────────
    context_labels: dict[str, set[str]] = defaultdict(set)

    for example in raw:
        clause_type = _extract_clause_type(example["question"])
        if clause_type is None:
            continue
        if _has_answer(example):
            context_labels[example["context"]].add(clause_type)

    # Also keep contexts where *no* clause was found (negative examples),
    # but limit them so the dataset isn't dominated by negatives.
    contexts_without_labels: list[str] = []
    for example in raw:
        ctx = example["context"]
        if ctx not in context_labels:
            contexts_without_labels.append(ctx)

    # De-duplicate negatives
    contexts_without_labels = list(set(contexts_without_labels))

    print(f"   ✅ Positive contexts (≥1 clause): {len(context_labels)}")
    print(f"   ✅ Negative contexts (no clause):  {len(contexts_without_labels)}")

    # ── Step 2: Build multi-hot label vectors ────────────────────────────
    texts: list[str] = []
    labels: list[list[float]] = []

    for context, clause_set in context_labels.items():
        multi_hot = [0.0] * NUM_LABELS
        for clause in clause_set:
            multi_hot[LABEL2ID[clause]] = 1.0
        texts.append(context)
        labels.append(multi_hot)

    # Add a balanced portion of negative examples
    neg_limit = min(len(contexts_without_labels), len(texts) // 3)
    for ctx in contexts_without_labels[:neg_limit]:
        texts.append(ctx)
        labels.append([0.0] * NUM_LABELS)

    print(f"   📊 Total classification examples: {len(texts)}")

    if max_samples and len(texts) > max_samples:
        texts = texts[:max_samples]
        labels = labels[:max_samples]
        print(f"   ⚠️  Capped to {max_samples} samples (dry-run mode)")

    # ── Step 3: Create HuggingFace Dataset and split ─────────────────────
    ds = Dataset.from_dict({"text": texts, "labels": labels})
    split = ds.train_test_split(test_size=test_size, seed=seed)

    return DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })


def tokenize_dataset(
    dataset_dict: DatasetDict,
    tokenizer: AutoTokenizer | None = None,
) -> DatasetDict:
    """
    Tokenize the classification dataset using the Legal-BERT tokenizer.

    Returns the same DatasetDict with added 'input_ids', 'attention_mask',
    and 'labels' columns — ready for the Trainer.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    def _tokenize(batch):
        encoding = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        # HuggingFace Trainer expects 'labels' as floats for BCEWithLogitsLoss
        encoding["labels"] = torch.tensor(batch["labels"], dtype=torch.float)
        return {k: v.tolist() for k, v in encoding.items()}

    tokenized = dataset_dict.map(
        _tokenize,
        batched=True,
        batch_size=64,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    tokenized.set_format("torch")
    return tokenized


# ── Quick Sanity Check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = build_classification_dataset(max_samples=100)
    print(f"\nTrain: {ds['train']}")
    print(f"Val:   {ds['validation']}")
    print(f"\nSample labels (first row): {ds['train'][0]['labels'][:10]}...")
