"""
Financial Analyzer Service

This service processes financial reports (PDF, TXT, CSV) and uses Gemini to
evaluate the overall financial risk, extract key risk factors, and suggest
mitigation strategies.
"""

from typing import List

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import (
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
)
from app.models.schemas import FinancialRiskAnalysis
from app.prompts.analysis_prompt import financial_risk_prompt
from app.services.document_loader import (
    load_files_from_uploads,
    combine_pages,
)
from app.services.vectorstore import create_vectorstore


def analyze_financial_report(uploaded_files: list) -> FinancialRiskAnalysis:
    """
    Full financial report analysis pipeline.

    Parameters
    ----------
    uploaded_files : list
        Streamlit uploaded file objects.

    Returns
    -------
    FinancialRiskAnalysis
        Parsed response from Gemini containing risk level, factors, and strategies.
    """
    if not uploaded_files:
        raise ValueError("Please upload at least one financial document.")

    # Load and preprocess documents
    pages = load_files_from_uploads(uploaded_files)
    if not pages:
        raise ValueError("Could not extract text from the uploaded documents.")

    full_text = combine_pages(pages)

    # We can also add financial reports to the vector store so users can chat with them
    create_vectorstore(pages)

    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
    )

    llm_with_structure = llm.with_structured_output(FinancialRiskAnalysis)
    chain = financial_risk_prompt | llm_with_structure

    # Pass the full text to Gemini for financial analysis
    result = chain.invoke({"document": full_text})

    return result
