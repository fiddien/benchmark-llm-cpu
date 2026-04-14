"""Structured output task — asks the model to respond with valid JSON."""
from __future__ import annotations

import json

from benchmark.tasks.base import BaseTask, TaskSample


_DEFAULT_SCHEMA_HINT = (
    "Respond ONLY with a valid JSON object matching the following schema:\n"
    "{schema}\n\n"
    "Do not include any explanation outside the JSON."
)


class StructuredOutputTask(BaseTask):
    """
    Evaluates whether the model can produce well-formed JSON that satisfies
    a given schema.  The ``sample.metadata`` dict may contain:

    - ``schema`` (str): JSON Schema string describing the expected output.
    - ``schema_hint`` (str): full prompt template (overrides the default).
    """

    name = "structured_output"

    def build_prompt(self, sample: TaskSample) -> str:
        schema = sample.metadata.get("schema", "{}")
        hint = sample.metadata.get("schema_hint", _DEFAULT_SCHEMA_HINT)
        base = hint.format(schema=schema)
        if sample.prompt:
            base = sample.prompt + "\n\n" + base
        return base

    def score(self, prediction: str, reference: str | None) -> dict[str, float]:
        # Primary metric: is the output parseable JSON?
        is_valid = 0.0
        field_match = 0.0
        try:
            parsed = json.loads(prediction)
            is_valid = 1.0
            # If a reference JSON is provided, compare keys and values
            if reference:
                ref = json.loads(reference)
                if isinstance(parsed, dict) and isinstance(ref, dict):
                    matched = sum(
                        1 for k, v in ref.items() if str(parsed.get(k, "")) == str(v)
                    )
                    field_match = matched / len(ref) if ref else 1.0
        except (json.JSONDecodeError, ValueError):
            pass
        metrics: dict[str, float] = {"valid_json": is_valid}
        if reference:
            metrics["field_match"] = field_match
        return metrics
