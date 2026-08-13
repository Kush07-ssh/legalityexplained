# ⚖️ LegalitySimplified

**Demystifying Legal Documents with AI** — Upload contracts, agreements, and policy documents to get instant clause classification, risk analysis, plain-English summaries, and interactive Q&A.

## 🏗️ Architecture

```
Document Upload → Text Extraction → Preprocessing
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
            BERT Classifier                              ChromaDB Vector Store
        (Clause Type Detection)                          (For Chat/RAG)
                    │                                           │
                    ▼                                           ▼
            Gemini (Risk Assessment                    Gemini (RAG Chat)
             + Explanation)                                     │
                    │                                           ▼
                    ▼                                   Interactive Q&A
        Risk Analysis Dashboard
                    │
                    ▼
            Gemini (Document Summary)
```

### Hybrid AI Pipeline
- **Legal-BERT** (fine-tuned on [CUAD](https://huggingface.co/datasets/theatticusproject/cuad)) → Classifies text into 41 legal clause types
- **Gemini 2.5 Flash** → Risk assessment, plain-English explanations, document summarization, and chat
- **ChromaDB + MiniLM-L6** → Vector store for RAG-based document Q&A

## 📁 Project Structure

```
Finance/
├── app/                          # Streamlit application
│   ├── main.py                   # UI entry point
│   ├── config.py                 # Centralized configuration
│   ├── models/schemas.py         # Pydantic schemas
│   ├── prompts/                  # LLM prompt templates
│   │   ├── analysis_prompt.py
│   │   ├── summary_prompt.py
│   │   └── chat_prompt.py
│   └── services/                 # Business logic
│       ├── analyzer.py           # Hybrid BERT+Gemini clause analysis
│       ├── summarizer.py         # Document summarization
│       ├── chat.py               # RAG chat service
│       ├── document_loader.py    # File loading & preprocessing
│       └── vectorstore.py        # ChromaDB management
├── bert/                         # BERT fine-tuning module
│   ├── fine_tune.py              # Training script
│   ├── classifier.py             # Inference wrapper
│   ├── dataset_utils.py          # CUAD dataset preprocessing
│   └── config.py                 # Training hyperparameters
├── models/                       # Saved fine-tuned models (gitignored)
├── database/                     # ChromaDB persistence (gitignored)
├── requirements.txt
├── run.py                        # Entry point
└── .env                          # API keys (gitignored)
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Fine-Tune BERT (Optional but Recommended)
```bash
# Full training (~30 min on GPU, ~2 hrs on CPU)
python -m bert.fine_tune

# Quick test run (100 samples, 1 epoch)
python -m bert.fine_tune --dry-run

# Custom training
python -m bert.fine_tune --epochs 3 --lr 3e-5
```

> **Note:** The app works without BERT — it falls back to Gemini-only mode. But BERT provides faster, more consistent clause classification.

### 4. Run the App
```bash
python run.py
# or
streamlit run app/main.py
```

## 🧠 BERT Fine-Tuning Details

| Setting | Value |
|---|---|
| Base Model | `nlpaueb/legal-bert-base-uncased` |
| Dataset | CUAD (510 contracts, 41 clause types) |
| Task | Multi-label classification |
| Loss | BCEWithLogitsLoss |
| LR Schedule | Cosine with warmup |
| Metrics | F1 (micro/macro), Precision, Recall |

### CUAD Clause Types (41)
Termination, Indemnification, Governing Law, Non-Compete, Exclusivity, IP Ownership, License Grant, Confidentiality, Revenue Sharing, Audit Rights, Liability Caps, and 30 more.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Clause Classification | Legal-BERT (fine-tuned) |
| Risk Assessment & Summary | Gemini 2.5 Flash |
| Vector Store | ChromaDB |
| Embeddings | MiniLM-L6-v2 |
| Framework | LangChain |
