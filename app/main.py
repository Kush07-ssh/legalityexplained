"""
LegalitySimplified — Streamlit Application

Main entry point for the web UI. Provides:
- Document upload (PDF/TXT)
- Clause classification & risk analysis (BERT + Gemini)
- Document summarization
- RAG-based document chat
- Financial Report Risk Management (Gemini)
"""

import sys
import os
# Ensure the root project directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
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
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #172554 0%, #2563eb 100%);
        color: white;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .gradient-text {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
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
st.markdown('<h1 class="gradient-text">⚖️ LegalitySimplified</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="color: #64748b; font-weight: 400;">Next-Gen Legal and Financial Risk Management Platform</h3>', unsafe_allow_html=True)

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

            # ── Graph: Risk Breakdown ──
            risk_counts = {"High": 0, "Medium": 0, "Low": 0}
            for row in st.session_state.all_rows:
                r_level = row.get("Risk Level", "Unknown")
                if r_level in risk_counts:
                    risk_counts[r_level] += 1
            
            # Draw bar chart
            fig_bar = px.bar(
                x=list(risk_counts.keys()), 
                y=list(risk_counts.values()),
                labels={'x': 'Risk Severity', 'y': 'Number of Clauses'},
                title="Clause Risk Breakdown",
                color=list(risk_counts.keys()),
                color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
            )
            fig_bar.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)

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

                if risk == "High":
                    color = "#ef4444"
                    bg_color = "#fef2f2"
                elif risk == "Medium":
                    color = "#f59e0b"
                    bg_color = "#fffbeb"
                else:
                    color = "#10b981"
                    bg_color = "#ecfdf5"

                st.markdown(f"""
                <div style="border-left: 6px solid {color}; padding: 18px; margin-bottom: 20px; background-color: {bg_color}; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); transition: transform 0.2s ease;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid rgba(0,0,0,0.05); padding-bottom: 8px;">
                        <h4 style="margin: 0; color: #1e3a8a; font-size: 20px; font-weight: 700;">{clause_name}</h4>
                        <span style="background-color: {color}; color: white; padding: 6px 14px; border-radius: 20px; font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; box-shadow: 0 2px 4px {color}40;">{risk} Risk</span>
                    </div>
                    <div style="color: #1f2937;">
                        <p style="margin: 8px 0; font-size: 15px;"><strong>🏷️ Classification:</strong> <span style="background-color: rgba(0,0,0,0.05); padding: 3px 8px; border-radius: 4px;">{clause_type}</span> <span style="font-size: 0.9em; color: #6b7280;">({confidence})</span></p>
                        <p style="margin: 12px 0 0 0; font-size: 16px; line-height: 1.6;"><strong>⚠️ Detailed Analysis:</strong><br/>{explanation}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

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
        
        col_score, col_summary = st.columns([1, 2])
        
        with col_score:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = result.risk_score_100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Financial Risk Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#1e3a8a"},
                    'steps': [
                        {'range': [0, 30], 'color': "#ecfdf5"},
                        {'range': [30, 70], 'color': "#fffbeb"},
                        {'range': [70, 100], 'color': "#fef2f2"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': result.risk_score_100
                    }
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
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
            for r in result.risk_factors:
                if r.severity in ["High", "Critical"]:
                    color = "#ef4444" # red
                    bg_color = "#fef2f2"
                elif r.severity == "Medium":
                    color = "#f59e0b" # yellow/orange
                    bg_color = "#fffbeb"
                else:
                    color = "#10b981" # green
                    bg_color = "#ecfdf5"
                
                st.markdown(f"""
                <div style="border-left: 6px solid {color}; padding: 18px; margin-bottom: 20px; background-color: {bg_color}; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); transition: transform 0.2s ease;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid rgba(0,0,0,0.05); padding-bottom: 8px;">
                        <h4 style="margin: 0; color: #1e3a8a; font-size: 18px; font-weight: 700;">{r.factor_name}</h4>
                        <span style="background-color: {color}; color: white; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; box-shadow: 0 2px 4px {color}40;">{r.severity} Risk</span>
                    </div>
                    <div style="color: #1f2937;">
                        <p style="margin: 8px 0; font-size: 15px;"><strong>📉 Affected Metric:</strong> <span style="background-color: rgba(0,0,0,0.05); padding: 3px 8px; border-radius: 4px;">{r.affected_metric}</span></p>
                        <p style="margin: 12px 0 0 0; font-size: 16px; line-height: 1.6;"><strong>⚠️ Details:</strong><br/>{r.description}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
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
