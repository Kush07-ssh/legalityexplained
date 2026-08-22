"""
Analysis Prompt — Risk Assessment for BERT-Classified Clauses

This prompt receives clause text WITH a clause type already identified by BERT.
Gemini's job is now focused: assess risk and explain in plain English.

When BERT is not available (model not yet trained), falls back to asking
Gemini to also identify the clause type.
"""

from langchain_core.prompts import PromptTemplate


# ── Primary Prompt (BERT-assisted) ───────────────────────────────────────────
# Used when BERT has already classified the clause type
ANALYSIS_WITH_BERT_TEMPLATE = """\
You are a legal risk analyst. A clause classification model has already
identified the clause type. Your job is to assess the risk and explain clearly.

### Clause Type (identified by classifier): {clause_type}
### Clause Text:
{clause_text}

### Instructions:
1. Assess the **risk level**: only "Low", "Medium", or "High".
2. Explain briefly and clearly in 40-50 words why you assigned this risk level.
3. Consider factors like: enforceability, one-sidedness, potential for disputes,
   financial exposure, and how common/standard this clause wording is.

Return the result strictly in the format defined by the provided schema.
Do not include any text outside the schema.
"""

analysis_with_bert_prompt = PromptTemplate(
    template=ANALYSIS_WITH_BERT_TEMPLATE,
    input_variables=["clause_type", "clause_text"],
)


# ── Fallback Prompt (no BERT) ────────────────────────────────────────────────
# Used when BERT model hasn't been fine-tuned yet — Gemini does everything
ANALYSIS_FALLBACK_TEMPLATE = """\
You are a legal assistant specializing in simplifying legal documents.
You will analyze the following document and segregate the clauses and assign it a risk level with detailed explanation.

### Instructions:
1. Identify the main purpose of the clause in simple English from the given document.
2. Assess the **risk level**: only "Low", "Medium", or "High".
3. Explain briefly and clearly with 40-50 words explanation why you assigned this risk level.

Return the result strictly in the format defined by the provided schema.
Do not include any text outside the schema.

Document to analyze:
{document}
"""

analysis_fallback_prompt = PromptTemplate(
    template=ANALYSIS_FALLBACK_TEMPLATE,
    input_variables=["document"],
)

# ── Financial Risk Prompt ────────────────────────────────────────────────────
FINANCIAL_RISK_TEMPLATE = """\
You are an expert financial analyst and risk manager. Analyze the following financial report data
and assess the current risk level, calculate a numerical risk score, identify key risk factors, and provide actionable mitigation strategies.

### Instructions:
1. Assess the **overall_risk_level**: only "Low", "Medium", or "High".
2. Calculate a **risk_score_100**: an integer from 1 to 100 representing the risk of failure (100 is extremely high risk).
3. Provide a brief **summary_explanation** (50-100 words).
4. Identify up to 5 key **risk_factors**. For each, provide the name, severity, detailed description, and the affected financial metric.
5. Recommend 3-5 practical **mitigation_strategies**. For each, provide the strategy name, description, expected impact, and implementation timeframe.

Return the result strictly in the format defined by the provided schema.
Do not include any text outside the schema.

Financial Report Data:
{document}
"""

financial_risk_prompt = PromptTemplate(
    template=FINANCIAL_RISK_TEMPLATE,
    input_variables=["document"],
)
