"""
Analyzer Service — Hybrid BERT + Gemini Pipeline

This is the core analysis engine:
1. BERT classifies each text chunk into clause types
2. Gemini assesses risk level and generates plain-English explanations

Falls back to Gemini-only mode if the BERT model hasn't been trained yet.
"""

import os
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    BERT_MODEL_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from app.models.schemas import DocumentAnalysis
from app.prompts.analysis_prompt import (
    analysis_with_bert_prompt,
    analysis_fallback_prompt,
)
from app.services.document_loader import (
    load_files_from_uploads,
    combine_pages,
)
from app.services.vectorstore import create_vectorstore
from app.services.summarizer import summarize


def _is_bert_available() -> bool:
    """Check if the fine-tuned BERT model exists on disk."""
    return os.path.exists(BERT_MODEL_DIR) and os.path.isfile(
        os.path.join(BERT_MODEL_DIR, "config.json")
    )


def _load_bert_classifier():
    """Lazy-load the BERT classifier only when needed."""
    from bert.classifier import ClauseClassifier
    return ClauseClassifier(model_path=BERT_MODEL_DIR)


def _analyze_with_bert(
    pages: List[Document],
    llm,
) -> List[dict]:
    """
    Hybrid pipeline: BERT classifies clause types → Gemini assesses risk.

    1. Split document into chunks
    2. BERT classifies each chunk into clause type(s)
    3. For each identified clause, Gemini generates risk + explanation
    """
    classifier = _load_bert_classifier()

    # Split into chunks for BERT classification
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(pages)

    all_rows = []
    seen_clauses = set()  # Avoid duplicate clause assessments

    for chunk in chunks:
        text = chunk.page_content.strip()
        if not text:
            continue

        # BERT classification
        predictions = classifier.predict(text)

        # Collect tasks for Gemini
        tasks = []
        for clause_type, confidence in predictions:
            if clause_type in seen_clauses:
                continue
            seen_clauses.add(clause_type)
            tasks.append((clause_type, confidence))
            
        if not tasks:
            continue

        # Execute Gemini API calls concurrently
        import concurrent.futures

        def assess_clause(c_type, conf, c_text):
            try:
                llm_with_structure = llm.with_structured_output(DocumentAnalysis)
                chain = analysis_with_bert_prompt | llm_with_structure
                res = chain.invoke({
                    "clause_type": c_type,
                    "clause_text": c_text,
                })
                
                rows = []
                for clause in res.clauses:
                    rows.append({
                        "Clause": clause.clause,
                        "Clause Type": c_type,
                        "Confidence": f"{conf:.0%}",
                        "Risk Level": clause.risk_level,
                        "Detailed Explanation": clause.detailed_explanation,
                    })
                return rows
            except Exception as e:
                return [{
                    "Clause": c_type,
                    "Clause Type": c_type,
                    "Confidence": f"{conf:.0%}",
                    "Risk Level": "Unknown",
                    "Detailed Explanation": f"Risk assessment failed: {str(e)}",
                }]

        # Use max_workers=3 to avoid hitting Gemini free-tier rate limits (HTTP 429)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_clause = {
                executor.submit(assess_clause, c_type, conf, text): c_type 
                for c_type, conf in tasks
            }
            for future in concurrent.futures.as_completed(future_to_clause):
                all_rows.extend(future.result())

    return all_rows


def _analyze_fallback(
    docs_to_analyze: List[Document],
    llm,
) -> List[dict]:
    """
    Fallback: Gemini does everything (same as original behavior).
    Used when BERT model hasn't been trained yet.
    """
    llm_with_structure = llm.with_structured_output(DocumentAnalysis)
    chain = analysis_fallback_prompt | llm_with_structure

    all_rows = []
    for doc in docs_to_analyze:
        if not doc.page_content:
            continue

        result = chain.invoke({"document": doc.page_content})

        for clause in result.clauses:
            all_rows.append({
                "Clause": clause.clause,
                "Clause Type": "—",
                "Confidence": "—",
                "Risk Level": clause.risk_level,
                "Detailed Explanation": clause.detailed_explanation,
            })

    return all_rows


def analyze(uploaded_files: list) -> Tuple[List[dict], str]:
    """
    Full document analysis pipeline.

    Parameters
    ----------
    uploaded_files : list
        Streamlit uploaded file objects.

    Returns
    -------
    (all_rows, summary)
        all_rows: List of clause analysis dicts
        summary: Formatted document summary string
    """
    if not uploaded_files:
        raise ValueError("Please upload at least one document.")

    # Load and preprocess documents
    pages = load_files_from_uploads(uploaded_files)
    if not pages:
        return [], ""

    full_text = combine_pages(pages)
    docs_for_analysis = [Document(page_content=full_text)]

    # Create vector store for chat feature
    create_vectorstore(pages)

    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
    )

    # Choose pipeline based on BERT availability
    use_bert = _is_bert_available()

    if use_bert:
        print("🧠 Using BERT + Gemini hybrid pipeline")
        all_rows = _analyze_with_bert(pages, llm)
    else:
        print("⚠️  BERT model not found — using Gemini-only fallback")
        print(f"   Train BERT with: python -m bert.fine_tune")
        all_rows = _analyze_fallback(docs_for_analysis, llm)

    # Generate summary (always uses Gemini)
    summary = summarize(full_text)

    return all_rows, summary
