"""
Chat Service — RAG-Based Legal Q&A

Handles document-grounded chat with conversation history.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import LLM_MODEL_NAME, LLM_TEMPERATURE
from app.prompts.chat_prompt import rag_prompt
from app.services.vectorstore import get_vectorstore


def format_docs_as_context(docs) -> str:
    """Format retrieved documents into a plain-text context string."""
    parts = []
    for i, d in enumerate(docs, start=1):
        page = getattr(d, "metadata", {}).get("page", None)
        header = f"[Source: page {page}]" if page is not None else f"[Source: {i}]"
        parts.append(f"{header}\n{d.page_content.strip()}")
    return "\n\n".join(parts)


def chat(query: str, chat_history: list) -> str:
    """
    Process a user query against the document vector store.

    Parameters
    ----------
    query : str
        The user's question.
    chat_history : list
        List of HumanMessage/AIMessage objects.

    Returns
    -------
    str
        The assistant's response.
    """
    vector_store = get_vectorstore()
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    docs = retriever.get_relevant_documents(query)
    context_text = format_docs_as_context(docs)

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
    )

    rag_chain = rag_prompt | llm | StrOutputParser()
    result = rag_chain.invoke({
        "context": context_text,
        "question": query,
        "chat_history": chat_history,
    })

    return result
