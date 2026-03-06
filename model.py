import os
import pandas as pd
import tempfile
from pathlib import Path
import re
from typing import List
import os

from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import load_prompt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from Prompts.chat_prompt import rag_prompt
from dotenv import load_dotenv
from Schemas.Clause import DocumentSummary

load_dotenv()

try:
    analysis_prompt = load_prompt("Prompts/analysis.json")
    summary_prompt = load_prompt("Prompts/summary_prompt.json")
    rag_prompt = rag_prompt
except FileNotFoundError:
    print("Error: not found. Please ensure the prompt file exists.")
    exit()

API_KEY = os.getenv("GOOGLE_API_KEY")
PERSIST_DIR = "./database"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

llm = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash',
    temperature=0.5
)
llm_with_structure = llm.with_structured_output(DocumentSummary)
def load_files_from_uploads(uploaded_files: list) -> List[Document]:
    """Loads content from a list of uploaded files (PDFs or TXTs)."""
    all_docs = []
    for file in uploaded_files:
        suffix = Path(file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            if suffix == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
                all_docs.extend(loader.load())
            elif suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
                all_docs.extend(loader.load())
        finally:
            os.remove(tmp_path)  # Ensure temporary file is always deleted

    return all_docs


def preprocess_text(text: str) -> str:
    """Cleans and normalizes legal document text."""
    text = text.lower()
    text = re.sub(r'page\s*\d+\s*(of\s*\d+)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)  # Consolidate whitespace
    text = text.replace("§", "section")
    # Kept this minimal to avoid removing important symbols in some contracts
    text = re.sub(r'[^a-zA-Z0-9.,;:!?()\-\'\"\s]', '', text)
    return text.strip()


def prepare_documents_and_vectorstore(uploaded_files=None):
    """
    Loads, processes, and chunks documents. Creates and persists a vector store
    if it doesn't exist, or loads it if it does.
    Returns both the processed documents and the vector store instance.
    """
    if not uploaded_files:
        raise ValueError("Please upload at least one document.")

    # 1. Load documents from uploaded files
    pages = load_files_from_uploads(uploaded_files)
    if not pages:
        return [], None

    full_text_content = "\n".join([preprocess_text(page.page_content) for page in pages])
    docs_for_analysis = [Document(page_content=full_text_content)]
    # Check if we need to build the vector store
    if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(pages)  # Split original pages for better retrieval context
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
        vectorstore.persist()
    else:

        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    return docs_for_analysis, vectorstore


def analyze(uploaded_files=None):
    """
    Analyzes the full content of uploaded documents to extract and evaluate legal clauses.
    """
    try:
        docs_to_analyze, vectorstore = prepare_documents_and_vectorstore(uploaded_files)
    except ValueError as e:
        print(e)
        return pd.DataFrame()  # Return empty dataframe if no files are uploaded


    all_rows = []
    content = ""
    chain1 = summary_prompt | llm

    for doc in docs_to_analyze:
        if not doc.page_content:
            continue
        content += doc.page_content
        chain = analysis_prompt | llm_with_structure
        result = chain.invoke({"document": doc.page_content})

        for clause in result.clauses:
            all_rows.append({
                "Clause": clause.clause,
                "Risk Level": clause.risk_level,
                "Detailed Explanation": clause.detailed_explanation
            })
    summary = chain1.invoke({"Agreement": content}).content
    return all_rows, summary

def format_docs_as_context(docs):
    """Simple formatter to make a plain-text context string from retrieved docs."""
    parts = []
    for i, d in enumerate(docs, start=1):
        page = getattr(d, "metadata", {}).get("page", None)
        header = f"[Source: page {page}]" if page is not None else f"[Source: {i}]"
        parts.append(f"{header}\n{d.page_content.strip()}")
    return "\n\n".join(parts)

def chat(query, chat_history):
    vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    docs = retriever.get_relevant_documents(query)
    context_text = format_docs_as_context(docs)

    rag_chain = rag_prompt | llm | StrOutputParser()
    result = rag_chain.invoke({
        "context": context_text,
        "question": query,
        "chat_history": chat_history   # <-- message objects, no normalize step
    })

    return result



