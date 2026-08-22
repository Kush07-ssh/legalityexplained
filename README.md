# ⚖️ LegalitySimplified & Financial Risk Manager

**Demystifying Legal and Financial Documents with AI** — Upload contracts, agreements, policies, and financial reports to get instant clause classification, financial risk assessment, plain-English summaries, and interactive Q&A.

## 🏗️ Architecture

```
Document Upload (Legal & Financial) → Text Extraction → Preprocessing
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
             (Legal & Financial)
                     │
                     ▼
             Gemini (Document Summary)
```

### Hybrid AI Pipeline
- **Legal-BERT** (fine-tuned on [CUAD](https://huggingface.co/datasets/theatticusproject/cuad)) → Classifies text into 41 legal clause types.
- **Gemini 2.5 Flash / 1.5 Pro** → Legal risk assessment, financial risk scoring, plain-English explanations, document summarization, mitigation strategies, and context-aware chat.
- **ChromaDB + MiniLM-L6** → Vector store for RAG-based document Q&A.

## 📁 Project Structure

```
Finance/
├── app/                          # Streamlit application
│   ├── main.py                   # UI entry point
│   ├── config.py                 # Centralized configuration
│   ├── models/schemas.py         # Pydantic schemas for Legal & Financial Risk
│   ├── prompts/                  # LLM prompt templates
│   │   ├── analysis_prompt.py
│   │   ├── summary_prompt.py
│   │   └── chat_prompt.py
│   └── services/                 # Business logic
│       ├── analyzer.py           # Hybrid BERT+Gemini legal clause analysis
│       ├── financial_analyzer.py # Financial risk assessment and strategy generation
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

> **Note:** The app works without BERT — it falls back to Gemini-only mode. But BERT provides faster, more consistent legal clause classification.

### 4. Run the App
```bash
python run.py
# or
streamlit run app/main.py
```

## 🌟 Core Features

### 📄 Legal Document Analysis
- **Automated Clause Detection**: Uses a fine-tuned BERT model to instantly identify up to 41 different legal clauses (Termination, Governing Law, Exclusivity, etc.).
- **Risk Assessment**: Gemini analyzes each clause to assign a risk severity (Low, Medium, High) and provides a 50-word plain-English explanation.
- **RAG Chatbot**: Chat directly with your contracts to ask specific questions about the context and risks.

### 📈 Financial Risk Management
- **Numerical Risk Scoring**: Upload financial reports (PDF, CSV, TXT) and receive an automated Risk Score (out of 100) indicating the overall financial health of the entity.
- **Key Risk Factors**: Automatically extracts a structured table of specific red flags (e.g., liquidity issues, debt-to-equity imbalances) along with their severities.
- **Mitigation Strategies**: Provides actionable, short-to-long-term strategies to improve financial stability and mitigate the identified risks.
- **Context-Aware Q&A**: The chatbot remembers your specific financial risks and summary, allowing you to ask detailed follow-up questions about mitigation implementation.

## 🧠 BERT Fine-Tuning Details

| Setting | Value |
|---|---|
| Base Model | `nlpaueb/legal-bert-base-uncased` |
| Dataset | CUAD (510 contracts, 41 clause types) |
| Task | Multi-label classification |
| Loss | BCEWithLogitsLoss |
| LR Schedule | Cosine with warmup |
| Metrics | F1 (micro/macro), Precision, Recall |

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Clause Classification | Legal-BERT (fine-tuned) |
| Risk Assessment & Summary | Gemini 1.5 Pro / 2.5 Flash |
| Vector Store | ChromaDB (langchain-chroma) |
| Embeddings | MiniLM-L6-v2 |
| Framework | LangChain |
