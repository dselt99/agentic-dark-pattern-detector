"""Pydantic schemas for structuring the agent's output.

This package provides the data models and utilities for enforcing
structured JSON output from the LLM agent.
"""

from .schemas import (
    PatternType,
    DetectedPattern,
    AuditResult,
    CartItem,
    ConsentStatus,
    InteractionSnapshot,
    JourneyLedger as JourneyLedgerSchema,
    AuditFlag,
)

from .utils import (
    get_json_schema,
    get_audit_result_schema,
    get_detected_pattern_schema,
    format_schema_for_openai,
    format_schema_for_anthropic,
    validate_and_parse,
    create_self_correction_prompt,
    inject_schema_into_system_prompt,
)

__all__ = [
    # Models
    "PatternType",
    "DetectedPattern",
    "AuditResult",
    "CartItem",
    "ConsentStatus",
    "InteractionSnapshot",
    "JourneyLedgerSchema",
    "AuditFlag",
    # Utilities
    "get_json_schema",
    "get_audit_result_schema",
    "get_detected_pattern_schema",
    "format_schema_for_openai",
    "format_schema_for_anthropic",
    "validate_and_parse",
    "create_self_correction_prompt",
    "inject_schema_into_system_prompt",
]
