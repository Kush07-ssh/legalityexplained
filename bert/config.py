"""
BERT Fine-Tuning Configuration

Hyperparameters, model identifiers, and CUAD clause label definitions
for fine-tuning Legal-BERT on contract clause classification.
"""

# ── Model ────────────────────────────────────────────────────────────────────
BASE_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
DATASET_NAME = "theatticusproject/cuad-qa"
OUTPUT_DIR = "models/legal-bert-clause-classifier"

# ── Training Hyperparameters ─────────────────────────────────────────────────
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_SEQ_LENGTH = 512
GRADIENT_ACCUMULATION_STEPS = 2

# A confidence score above this threshold counts as a positive prediction
PREDICTION_THRESHOLD = 0.5

# ── CUAD Clause Categories (41 types) ────────────────────────────────────────
# These are the 41 clause types annotated in the CUAD dataset.
# Order matters — index position maps to the model's output logits.
CLAUSE_LABELS = [
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "Competitive Restriction Exception",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
]

NUM_LABELS = len(CLAUSE_LABELS)

# Mapping from label name to index and vice-versa
LABEL2ID = {label: idx for idx, label in enumerate(CLAUSE_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(CLAUSE_LABELS)}
