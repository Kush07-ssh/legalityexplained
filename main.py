import streamlit as st
from model import analyze, chat
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(page_title="LegalitySimplified", layout="wide")

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.summary_text = ""
    st.session_state.all_rows = []
    st.session_state.message_history = []
    st.session_state.show_summary = False
    st.session_state.show_risk_analysis = False

st.header("Demystifying Legal Documents")
st.markdown("##### Upload your Legal Documents for enhanced explanations of clauses and legal terms.")

uploaded_files = st.file_uploader(
    label="Upload Legal Documents (PDF, TXT)",
    accept_multiple_files=True,
    type=['pdf', 'txt']
)

if uploaded_files and not st.session_state.analysis_done:
    with st.spinner("Analyzing documents... This may take a moment."):
        all_rows, summary = analyze(uploaded_files)
        st.session_state.summary_text = summary
        st.session_state.all_rows = all_rows
        st.session_state.analysis_done = True

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

    if st.session_state.show_summary:
        st.subheader("Document Summary")
        st.write(st.session_state.summary_text)

    if st.session_state.show_risk_analysis:
        st.subheader("Risk Level Analysis")
        risk_colors = {"High": "🔴 High Risk", "Medium": "🟡 Medium Risk", "Low": "🟢 Low Risk"}
        for i, row in enumerate(st.session_state.all_rows, start=1):
            risk = row.get("Risk Level", "Unknown")
            risk_display = risk_colors.get(risk, "⚪ Unknown")
            clause_name = row.get("Clause", "Unnamed Clause")
            explanation = row.get("Detailed Explanation", "No explanation available.")

            with st.expander(f"{clause_name} ({risk_display})"):
                st.markdown(f"**Explanation:** {explanation}")


# --- 5. Chat section ---
if st.session_state.analysis_done:
    st.divider()
    st.subheader("Chat About Your Document")

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


    if user_input := st.chat_input("Type your query here..."):
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Thinking..."):
            assistant_response = chat(
                query=user_input,
                chat_history=st.session_state.message_history
            )

        with st.chat_message("assistant"):
            st.write(assistant_response)

        st.session_state.message_history.append(HumanMessage(content=user_input))
        st.session_state.message_history.append(AIMessage(content=assistant_response))
