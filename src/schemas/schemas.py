"""Pydantic schemas for structuring the agent's output.

This module defines the strict data models that enforce JSON output protocols,
ensuring deterministic, structured data from the probabilistic LLM.
"""

from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import Optional, List
from datetime import datetime


class PatternType(str, Enum):
    """Enumeration of dark pattern types based on SKILL.md definitions."""

    ROACH_MOTEL = "roach_motel"
    FALSE_URGENCY = "false_urgency"
    CONFIRMSHAMING = "confirmshaming"
    SNEAK_INTO_BASKET = "sneak_into_basket"
    FORCED_CONTINUITY = "forced_continuity"


class DetectedPattern(BaseModel):
    """Represents a single instance of a suspected dark pattern.

    This model captures the what, where, and why of a detected pattern,
    ensuring the agent provides structured evidence rather than free-form text.
    """

    pattern_type: PatternType = Field(
        ...,
        description="The classification of the dark pattern based on definitions in SKILL.md.",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="A normalized score (0-1) representing the agent's certainty.",
    )
    element_selector: str = Field(
        ...,
        description="The precise CSS or XPath selector of the DOM element involved.",
    )
    reasoning: str = Field(
        ...,
        description=(
            "A concise explanation of WHY this fits the definition. "
            "Must reference specific heuristics (e.g., 'Timer reset on reload')."
        ),
    )
    evidence: Optional[str] = Field(
        None,
        description=(
            "The text content or attribute value that triggered the detection "
            "(e.g., 'No, I hate saving money')."
        ),
    )

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence score meets minimum threshold for reporting."""
        if v < 0.7:
            raise ValueError(
                "Confidence score too low. Do not report uncertainty as a finding."
            )
        return v


class AuditResult(BaseModel):
    """Encapsulates the entire audit session, aggregating all findings and metadata.

    This is the primary output structure that the agent must produce,
    ensuring all audit results are consistently formatted and validated.
    """

    target_url: str = Field(..., description="The URL that was audited.")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the audit was performed.",
    )
    findings: List[DetectedPattern] = Field(
        default_factory=list,
        description="List of detected dark patterns. Empty if none found.",
    )
    screenshot_paths: List[str] = Field(
        default_factory=list,
        description="Relative paths to screenshot evidence files in artifacts/ directory.",
    )
    summary: str = Field(
        ...,
        description="An executive summary of the audit for a human reader.",
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "target_url": "https://dark-shop.com",
                "findings": [],
            }
        }
