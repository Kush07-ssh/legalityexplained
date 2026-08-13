"""
LegalitySimplified — Streamlit Application

Main entry point for the web UI. Provides:
- Document upload (PDF/TXT)
- Clause classification & risk analysis (BERT + Gemini)
- Document summarization
- RAG-based document chat
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from app.services.analyzer import analyze
from app.services.chat import chat


# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="LegalitySimplified",
    page_icon="⚖️",
    layout="wide",
)

# ── Session State Initialization ─────────────────────────────────────────────
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.summary_text = ""
    st.session_state.all_rows = []
    st.session_state.message_history = []
    st.session_state.show_summary = False
    st.session_state.show_risk_analysis = False

# ── Header ───────────────────────────────────────────────────────────────────
st.header("⚖️ Demystifying Legal Documents")
st.markdown(
    "##### Upload your Legal Documents for enhanced explanations of clauses and legal terms."
)

# ── File Upload ──────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    label="Upload Legal Documents (PDF, TXT)",
    accept_multiple_files=True,
    type=["pdf", "txt"],
)

# ── Analysis ─────────────────────────────────────────────────────────────────
if uploaded_files and not st.session_state.analysis_done:
    with st.spinner("Analyzing documents... This may take a moment."):
        all_rows, summary = analyze(uploaded_files)
        st.session_state.summary_text = summary
        st.session_state.all_rows = all_rows
        st.session_state.analysis_done = True

# ── Results Display ──────────────────────────────────────────────────────────
if st.session_state.analysis_done:
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 Show Document Summary", use_container_width=True):
            st.session_state.show_summary = not st.session_state.show_summary
            st.session_state.show_risk_analysis = False

    with col2:
        if st.button("🔍 Show Risk Level Analysis", use_container_width=True):
            st.session_state.show_risk_analysis = not st.session_state.show_risk_analysis
            st.session_state.show_summary = False

    # ── Summary View ─────────────────────────────────────────────────────
    if st.session_state.show_summary:
        st.subheader("📄 Document Summary")
        st.write(st.session_state.summary_text)

    # ── Risk Analysis View ───────────────────────────────────────────────
    if st.session_state.show_risk_analysis:
        st.subheader("🔍 Risk Level Analysis")

        risk_colors = {
            "High": "🔴 High Risk",
            "Medium": "🟡 Medium Risk",
            "Low": "🟢 Low Risk",
        }

        for i, row in enumerate(st.session_state.all_rows, start=1):
            risk = row.get("Risk Level", "Unknown")
            risk_display = risk_colors.get(risk, "⚪ Unknown")
            clause_name = row.get("Clause", "Unnamed Clause")
            clause_type = row.get("Clause Type", "—")
            confidence = row.get("Confidence", "—")
            explanation = row.get("Detailed Explanation", "No explanation available.")

            # Show clause type + confidence if from BERT
            header = f"{clause_name} ({risk_display})"
            if clause_type != "—":
                header += f" — 🏷️ {clause_type} [{confidence}]"

            with st.expander(header):
                if clause_type != "—":
                    st.markdown(f"**Clause Type (BERT):** {clause_type}")
                    st.markdown(f"**Classification Confidence:** {confidence}")
                st.markdown(f"**Risk Level:** {risk_display}")
                st.markdown(f"**Explanation:** {explanation}")


# ── Chat Section ─────────────────────────────────────────────────────────────
if st.session_state.analysis_done:
    st.divider()
    st.subheader("💬 Chat About Your Document")

    # Render chat history
    for msg in st.session_state.message_history:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            continue
        with st.chat_message(role):
            st.write(msg.content)

    # Chat input
    if user_input := st.chat_input("Type your query here..."):
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Thinking..."):
            assistant_response = chat(
                query=user_input,
                chat_history=st.session_state.message_history,
            )

        with st.chat_message("assistant"):
            st.write(assistant_response)

        st.session_state.message_history.append(HumanMessage(content=user_input))
        st.session_state.message_history.append(AIMessage(content=assistant_response))
