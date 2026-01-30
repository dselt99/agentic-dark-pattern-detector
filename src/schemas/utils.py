"""Schema injection utilities for LLM integration.

This module implements the injection strategy for Pillar 3, converting Pydantic
models to JSON Schema and handling validation errors with self-correction loops.
"""

import json
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel, ValidationError

from .schemas import AuditResult, DetectedPattern


def get_json_schema(model_class: Type[BaseModel]) -> Dict[str, Any]:
    """Convert a Pydantic model to JSON Schema.

    Args:
        model_class: The Pydantic model class to convert.

    Returns:
        JSON Schema dictionary compatible with LLM function calling.
    """
    return model_class.model_json_schema()


def get_audit_result_schema() -> Dict[str, Any]:
    """Get JSON Schema for AuditResult model.

    Returns:
        JSON Schema for AuditResult, ready for LLM injection.
    """
    return get_json_schema(AuditResult)


def get_detected_pattern_schema() -> Dict[str, Any]:
    """Get JSON Schema for DetectedPattern model.

    Returns:
        JSON Schema for DetectedPattern, ready for LLM injection.
    """
    return get_json_schema(DetectedPattern)


def format_schema_for_openai(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Format JSON Schema for OpenAI function calling API.

    OpenAI expects a specific format with 'name', 'description', and 'parameters'.

    Args:
        schema: The JSON Schema dictionary.

    Returns:
        Formatted schema for OpenAI function calling.
    """
    return {
        "name": "audit_result",
        "description": "The complete audit result with detected dark patterns.",
        "parameters": schema,
    }


def format_schema_for_anthropic(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Format JSON Schema for Anthropic Claude API.

    Anthropic uses structured outputs with JSON Schema in a different format.

    Args:
        schema: The JSON Schema dictionary.

    Returns:
        Formatted schema for Anthropic structured outputs.
    """
    # Anthropic uses the schema directly but may need adjustments
    # This is a placeholder - actual format depends on Anthropic API version
    return {
        "type": "object",
        "schema": schema,
    }


def validate_and_parse(
    json_str: str, model_class: Type[BaseModel] = AuditResult
) -> tuple[Optional[BaseModel], Optional[str]]:
    """Validate JSON string against Pydantic model.

    This implements the self-correction mechanism: if validation fails,
    the error message can be fed back to the LLM for retry.

    Args:
        json_str: JSON string from LLM output.
        model_class: Pydantic model class to validate against.

    Returns:
        Tuple of (parsed_model, error_message).
        If valid: (model_instance, None)
        If invalid: (None, error_description)
    """
    try:
        # Parse JSON string
        data = json.loads(json_str)

        # Validate against Pydantic model
        instance = model_class.model_validate(data)
        return instance, None

    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON format: {str(e)}"
        return None, error_msg

    except ValidationError as e:
        # Format Pydantic validation errors for LLM feedback
        errors = []
        for error in e.errors():
            field = " -> ".join(str(loc) for loc in error["loc"])
            error_type = error["type"]
            error_msg = error.get("msg", "")
            errors.append(f"Field '{field}': {error_type} - {error_msg}")

        error_description = "Validation errors:\n" + "\n".join(errors)
        return None, error_description

    except Exception as e:
        error_msg = f"Unexpected error during validation: {str(e)}"
        return None, error_msg


def create_self_correction_prompt(
    original_prompt: str, validation_error: str, attempt: int, max_attempts: int = 3
) -> str:
    """Create a self-correction prompt for the LLM.

    When validation fails, this prompt instructs the LLM to fix the output.

    Args:
        original_prompt: The original prompt that generated the invalid output.
        validation_error: The validation error message.
        attempt: Current attempt number (1-indexed).
        max_attempts: Maximum number of retry attempts.

    Returns:
        Enhanced prompt with validation error feedback.
    """
    if attempt >= max_attempts:
        return (
            f"{original_prompt}\n\n"
            f"CRITICAL: Previous attempts failed validation. "
            f"This is attempt {attempt}/{max_attempts}. "
            f"Please carefully review the schema requirements and ensure all fields are correct.\n\n"
            f"Validation errors from previous attempt:\n{validation_error}\n\n"
            f"Please provide a valid response that matches the schema exactly."
        )

    return (
        f"{original_prompt}\n\n"
        f"VALIDATION ERROR (Attempt {attempt}/{max_attempts}):\n"
        f"The previous response did not match the required schema. "
        f"Please correct the following errors and try again:\n\n"
        f"{validation_error}\n\n"
        f"Please provide a corrected response that matches the schema exactly."
    )


def inject_schema_into_system_prompt(
    base_prompt: str, schema: Dict[str, Any], provider: str = "anthropic"
) -> str:
    """Inject JSON Schema into system prompt.

    This creates a system prompt that includes the schema requirements,
    forcing the LLM to output structured data.

    Args:
        base_prompt: The base system prompt (e.g., from SKILL.md).
        schema: JSON Schema dictionary.
        provider: LLM provider ("openai" or "anthropic").

    Returns:
        Enhanced system prompt with schema requirements.
    """
    schema_str = json.dumps(schema, indent=2)

    schema_instruction = f"""
You MUST output your response as a valid JSON object that matches the following schema exactly:

```json
{schema_str}
```

Important constraints:
- All required fields must be present
- confidence_score must be between 0.0 and 1.0, and must be >= 0.7 for any reported finding
- element_selector must be a valid CSS or XPath selector
- reasoning must reference specific heuristics from the skill definitions
- Do not output free-form text; only output valid JSON matching this schema
"""

    return f"{base_prompt}\n\n{schema_instruction}"
