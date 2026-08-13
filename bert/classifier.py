"""
Clause Classifier — Inference Wrapper

Loads the fine-tuned Legal-BERT model and provides a clean API for
classifying legal text into clause types.

Usage:
    from bert.classifier import ClauseClassifier

    clf = ClauseClassifier()                        # loads from default path
    clf = ClauseClassifier("path/to/custom/model")  # or a custom path

    results = clf.predict("This agreement may be terminated by either party...")
    # → [("Termination For Convenience", 0.92), ("Non-Compete", 0.73)]

    batch = clf.predict_batch(["clause 1 text", "clause 2 text"])
"""

import os
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from bert.config import (
    OUTPUT_DIR,
    MAX_SEQ_LENGTH,
    PREDICTION_THRESHOLD,
    ID2LABEL,
    NUM_LABELS,
)


class ClauseClassifier:
    """
    Wrapper around the fine-tuned Legal-BERT model for clause classification.

    The model outputs multi-label predictions — a single text chunk can be
    classified into multiple clause types simultaneously.
    """

    def __init__(
        self,
        model_path: str = OUTPUT_DIR,
        threshold: float = PREDICTION_THRESHOLD,
        device: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        model_path : str
            Path to the fine-tuned model directory.
        threshold : float
            Confidence threshold for positive predictions (0.0–1.0).
        device : str | None
            Force a specific device ('cpu', 'cuda', 'mps').
            Auto-detects if None.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Fine-tuned model not found at '{model_path}'. "
                f"Run 'python -m bert.fine_tune' first to train the model."
            )

        self.threshold = threshold

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ ClauseClassifier loaded from '{model_path}' on {self.device}")

    def predict(self, text: str) -> list[tuple[str, float]]:
        """
        Classify a single text into clause types.

        Parameters
        ----------
        text : str
            The legal text (clause, paragraph, or chunk) to classify.

        Returns
        -------
        list of (clause_type, confidence) tuples, sorted by confidence
        descending. Only includes predictions above the threshold.
        """
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits[0]
            probs = torch.sigmoid(logits).cpu().numpy()

        results = []
        for idx, prob in enumerate(probs):
            if prob >= self.threshold:
                results.append((ID2LABEL[idx], float(prob)))

        # Sort by confidence descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def predict_batch(self, texts: list[str]) -> list[list[tuple[str, float]]]:
        """
        Classify multiple texts in a single forward pass (more efficient).

        Parameters
        ----------
        texts : list[str]
            List of legal text chunks.

        Returns
        -------
        list of prediction lists (one per input text).
        """
        encoding = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)
            all_probs = torch.sigmoid(outputs.logits).cpu().numpy()

        batch_results = []
        for probs in all_probs:
            results = []
            for idx, prob in enumerate(probs):
                if prob >= self.threshold:
                    results.append((ID2LABEL[idx], float(prob)))
            results.sort(key=lambda x: x[1], reverse=True)
            batch_results.append(results)

        return batch_results

    def predict_top_label(self, text: str) -> tuple[str, float]:
        """
        Get the single most likely clause type for a text.
        Useful when you only need the primary classification.

        Returns ("Unknown", 0.0) if no clause exceeds the threshold.
        """
        preds = self.predict(text)
        if preds:
            return preds[0]
        return ("Unknown", 0.0)


# ── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    clf = ClauseClassifier()

    test_texts = [
        "This agreement may be terminated by either party with 30 days written notice.",
        "The governing law of this contract shall be the laws of the State of Delaware.",
        "Neither party shall solicit or hire employees of the other party during the term.",
    ]

    for text in test_texts:
        preds = clf.predict(text)
        print(f"\n📄 Text: {text[:80]}...")
        if preds:
            for label, conf in preds:
                print(f"   → {label}: {conf:.2%}")
        else:
            print("   → No clause type detected above threshold")
