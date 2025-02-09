from typing import List

from pydantic import BaseModel, Field


class HeadlineAnalysis(BaseModel):
    """singled headline analysis model."""
    theme: str = Field(description="Theme or topic of the headline")
    key_points: List[str] = Field(description="Key insights and implications of each topic.")
    summary: List[str] = Field(description="Summary of the provided topic")
    relevant_links: List[str] = Field(description="Links or references relevant to the analysis.")
    priority_level: float = Field(
        description="Priority level of the headline. Higher values indicate higher priority.")


class AnalysisResult(BaseModel):
    """Structured analysis result model."""
    email_id: str = Field(description="Email ID of the analysis result.")
    headline_insights: List[HeadlineAnalysis] = Field(description="Key insights and implications of each headline.")

    def to_dict(self):
        _dict = self.model_dump(mode="json")
        return _dict
