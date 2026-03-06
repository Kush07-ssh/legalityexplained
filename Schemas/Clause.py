
from pydantic import BaseModel, Field
from typing import Literal, List

class Clause(BaseModel):
    clause: str = Field(..., description="Name or title of the legal clause")
    risk_level: Literal["Low", "Medium", "High"] = Field(..., description="Risk severity of this clause")
    detailed_explanation: str = Field(..., description="Simplified explanation of the clause")


class DocumentSummary(BaseModel):
    clauses: List[Clause] = Field(
        ..., description="List of clauses with their risk levels and explanations"
    )

