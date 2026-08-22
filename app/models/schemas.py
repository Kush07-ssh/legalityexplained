"""
Pydantic Schemas for Structured LLM Output

These schemas are used with Gemini's structured output mode to get
consistent, parseable responses for risk assessment and explanation.
"""

from pydantic import BaseModel, Field
from typing import Literal, List


class ClauseRiskAssessment(BaseModel):
    """Risk assessment for a single identified clause."""
    clause: str = Field(
        ...,
        description="Name or title of the legal clause (from BERT classification or LLM)"
    )
    risk_level: Literal["Low", "Medium", "High"] = Field(
        ...,
        description="Risk severity of this clause"
    )
    detailed_explanation: str = Field(
        ...,
        description="Simplified explanation of the clause and why this risk level was assigned"
    )


class DocumentAnalysis(BaseModel):
    """Complete analysis containing all clause risk assessments."""
    clauses: List[ClauseRiskAssessment] = Field(
        ...,
        description="List of clauses with their risk levels and explanations"
    )

class RiskFactor(BaseModel):
    factor_name: str = Field(..., description="Short name of the risk (e.g., Liquidity Crisis)")
    severity: Literal["Low", "Medium", "High", "Critical"] = Field(..., description="Severity of this specific risk")
    description: str = Field(..., description="Detailed explanation of the risk found in the report")
    affected_metric: str = Field(..., description="The financial metric or area affected (e.g., Cash Flow, Debt/Equity)")

class MitigationStrategy(BaseModel):
    strategy_name: str = Field(..., description="Short name of the strategy")
    description: str = Field(..., description="How to implement this strategy")
    expected_impact: Literal["Low", "Medium", "High"] = Field(..., description="Expected positive impact of this strategy")
    timeframe: str = Field(..., description="Timeframe for implementation (e.g., Short-term, Long-term, Immediate)")

class FinancialRiskAnalysis(BaseModel):
    """Overall financial risk assessment of a company."""
    overall_risk_level: Literal["Low", "Medium", "High"] = Field(
        ...,
        description="Overall financial risk severity"
    )
    risk_score_100: int = Field(
        ...,
        description="A calculated numerical risk score from 1 to 100, where 100 represents the highest possible risk of bankruptcy or failure."
    )
    summary_explanation: str = Field(
        ...,
        description="A brief executive summary explaining the risk level and overall financial health."
    )
    risk_factors: List[RiskFactor] = Field(
        ...,
        description="Key risk factors identified in the financial report"
    )
    mitigation_strategies: List[MitigationStrategy] = Field(
        ...,
        description="Recommended strategies to mitigate the identified risks and improve financial health"
    )
