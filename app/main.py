"""
LegalitySimplified — Streamlit Application

Main entry point for the web UI. Provides:
- Document upload (PDF/TXT)
- Clause classification & risk analysis (BERT + Gemini)
- Document summarization
- RAG-based document chat
- Financial Report Risk Management (Gemini)
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from app.services.analyzer import analyze
from app.services.chat import chat
from app.services.financial_analyzer import analyze_financial_report


# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="LegalitySimplified Risk Manager",
    layout="wide",
)

# Custom CSS for a more professional look
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #172554;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Initialization ─────────────────────────────────────────────
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.summary_text = ""
    st.session_state.all_rows = []
    st.session_state.message_history = []
    st.session_state.show_summary = False
    st.session_state.show_risk_analysis = False

if "financial_analysis_done" not in st.session_state:
    st.session_state.financial_analysis_done = False
    st.session_state.financial_result = None
    st.session_state.fin_message_history = []

# ── Header ───────────────────────────────────────────────────────────────────
st.title("LegalitySimplified")
st.subheader("Legal and Financial Risk Management Platform")

tab1, tab2 = st.tabs(["Legal Document Analysis", "Financial Risk Management"])

with tab1:
    st.markdown("Upload legal documents to extract clauses, evaluate risks, and summarize content.")

    # ── File Upload ──────────────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        label="Upload Legal Documents (Supported formats: PDF, TXT)",
        accept_multiple_files=True,
        type=["pdf", "txt"],
        key="legal_upload"
    )

    # ── Analysis ─────────────────────────────────────────────────────────────────
    if uploaded_files and not st.session_state.analysis_done:
        with st.spinner("Analyzing legal documents..."):
            all_rows, summary = analyze(uploaded_files)
            st.session_state.summary_text = summary
            st.session_state.all_rows = all_rows
            st.session_state.analysis_done = True

    # ── Results Display ──────────────────────────────────────────────────────────
    if st.session_state.analysis_done:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Show Document Summary", use_container_width=True):
                st.session_state.show_summary = not st.session_state.show_summary
                st.session_state.show_risk_analysis = False

        with col2:
            if st.button("Show Risk Level Analysis", use_container_width=True):
                st.session_state.show_risk_analysis = not st.session_state.show_risk_analysis
                st.session_state.show_summary = False

        # ── Summary View ─────────────────────────────────────────────────────
        if st.session_state.show_summary:
            st.markdown("### Document Summary")
            st.write(st.session_state.summary_text)

        # ── Risk Analysis View ───────────────────────────────────────────────
        if st.session_state.show_risk_analysis:
            st.markdown("### Risk Level Analysis")

            risk_colors = {
                "High": "High Risk",
                "Medium": "Medium Risk",
                "Low": "Low Risk",
            }

            for i, row in enumerate(st.session_state.all_rows, start=1):
                risk = row.get("Risk Level", "Unknown")
                risk_display = risk_colors.get(risk, "Unknown")
                clause_name = row.get("Clause", "Unnamed Clause")
                clause_type = row.get("Clause Type", "—")
                confidence = row.get("Confidence", "—")
                explanation = row.get("Detailed Explanation", "No explanation available.")

                header = f"{clause_name} - {risk_display}"
                if clause_type != "—":
                    header += f" (Type: {clause_type} | Confidence: {confidence})"

                with st.expander(header):
                    if clause_type != "—":
                        st.markdown(f"**Clause Type (BERT):** {clause_type}")
                        st.markdown(f"**Classification Confidence:** {confidence}")
                    st.markdown(f"**Risk Level:** {risk_display}")
                    st.markdown(f"**Explanation:** {explanation}")

    # ── Chat Section ─────────────────────────────────────────────────────────────
    if st.session_state.analysis_done:
        st.divider()
        st.markdown("### Document Q&A")

        for msg in st.session_state.message_history:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.write(msg.content)

        if user_input := st.chat_input("Enter your query regarding the legal document...", key="legal_chat"):
            with st.chat_message("user"):
                st.write(user_input)

            with st.spinner("Processing query..."):
                analysis_context = "Document Summary:\n" + st.session_state.summary_text + "\n\nIdentified Clause Risks:\n"
                for row in st.session_state.all_rows:
                    analysis_context += f"- {row.get('Clause', '')} ({row.get('Risk Level', '')} Risk): {row.get('Detailed Explanation', '')}\n"

                assistant_response = chat(
                    query=user_input,
                    chat_history=st.session_state.message_history,
                    analysis_context=analysis_context,
                )

            with st.chat_message("assistant"):
                st.write(assistant_response)

            st.session_state.message_history.append(HumanMessage(content=user_input))
            st.session_state.message_history.append(AIMessage(content=assistant_response))


with tab2:
    st.markdown("Upload financial reports to assess risk factors and generate mitigation strategies.")

    # ── Financial File Upload ────────────────────────────────────────────────────
    fin_uploaded_files = st.file_uploader(
        label="Upload Financial Reports (Supported formats: PDF, TXT, CSV)",
        accept_multiple_files=True,
        type=["pdf", "txt", "csv"],
        key="fin_upload"
    )

    if fin_uploaded_files and not st.session_state.financial_analysis_done:
        with st.spinner("Analyzing financial data and extracting risk factors..."):
            fin_result = analyze_financial_report(fin_uploaded_files)
            st.session_state.financial_result = fin_result
            st.session_state.financial_analysis_done = True

    if st.session_state.financial_analysis_done:
        st.markdown("### Financial Risk Assessment")
        
        result = st.session_state.financial_result
        
        col_score, col_risk, col_summary = st.columns([1, 1, 2])
        
        with col_score:
            st.metric(
                label="Risk Score (out of 100)",
                value=f"{result.risk_score_100} / 100",
                delta="High Risk" if result.risk_score_100 > 70 else ("Medium Risk" if result.risk_score_100 > 30 else "Low Risk"),
                delta_color="inverse"
            )

        with col_risk:
            st.metric(
                label="Overall Severity",
                value=result.overall_risk_level
            )
            
        with col_summary:
            st.markdown("**Executive Summary**")
            st.info(result.summary_explanation)
            
        st.divider()
        
        st.markdown("#### ⚠️ Key Risk Factors Identified")
        if result.risk_factors:
            risk_data = [
                {
                    "Risk Factor": r.factor_name,
                    "Severity": r.severity,
                    "Affected Metric": r.affected_metric,
                    "Description": r.description
                }
                for r in result.risk_factors
            ]
            st.table(risk_data)
            
        st.markdown("#### 💡 Recommended Mitigation Strategies")
        if result.mitigation_strategies:
            strat_data = [
                {
                    "Strategy": s.strategy_name,
                    "Impact": s.expected_impact,
                    "Timeframe": s.timeframe,
                    "Implementation": s.description
                }
                for s in result.mitigation_strategies
            ]
            st.table(strat_data)
            
        st.divider()
        st.markdown("### Financial Q&A")

        for msg in st.session_state.fin_message_history:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.write(msg.content)

        if fin_user_input := st.chat_input("Ask a question about the financial report...", key="fin_chat"):
            with st.chat_message("user"):
                st.write(fin_user_input)

            with st.spinner("Processing query..."):
                res = st.session_state.financial_result
                analysis_context = f"Overall Risk: {res.overall_risk_level} (Score: {res.risk_score_100}/100)\n"
                analysis_context += f"Executive Summary: {res.summary_explanation}\n"
                analysis_context += "Risk Factors: " + "; ".join([r.factor_name for r in res.risk_factors]) + "\n"
                analysis_context += "Mitigation Strategies: " + "; ".join([s.strategy_name for s in res.mitigation_strategies]) + "\n"

                fin_assistant_response = chat(
                    query=fin_user_input,
                    chat_history=st.session_state.fin_message_history,
                    analysis_context=analysis_context,
                )

            with st.chat_message("assistant"):
                st.write(fin_assistant_response)

            st.session_state.fin_message_history.append(HumanMessage(content=fin_user_input))
            st.session_state.fin_message_history.append(AIMessage(content=fin_assistant_response))
