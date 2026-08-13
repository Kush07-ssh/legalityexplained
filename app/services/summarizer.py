"""
Summarizer Service

Generates structured legal document summaries using Gemini.
"""

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import LLM_MODEL_NAME, LLM_TEMPERATURE
from app.prompts.summary_prompt import summary_prompt


def summarize(content: str) -> str:
    """
    Generate a structured summary of the legal document.

    Parameters
    ----------
    content : str
        The full preprocessed document text.

    Returns
    -------
    str
        Formatted summary with parties, terms, clause breakdowns, etc.
    """
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
    )

    chain = summary_prompt | llm
    result = chain.invoke({"Agreement": content})
    return result.content
