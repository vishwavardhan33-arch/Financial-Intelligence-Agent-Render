"""Resolve a plan step's args (literals or ValueRefs) against the real,
already-computed results of earlier steps.

This is where the "never let the LLM re-type numbers" guarantee is
actually enforced at the agent level: a ValueRef never contains a number
itself, only a pointer (step index + field name) that this code resolves
against the real stored output of that step.
"""
from __future__ import annotations


class ResolutionError(Exception):
    pass


def resolve_args(args: dict, step_results: list[dict]) -> dict[str, float]:
    resolved: dict[str, float] = {}
    for name, value in args.items():
        if isinstance(value, dict) and "from_step" in value:
            resolved[name] = _resolve_ref(value, step_results)
        else:
            resolved[name] = value
    return resolved


def _resolve_ref(ref: dict, step_results: list[dict]) -> float:
    step_idx = ref["from_step"]
    field = ref["field"]
    row = ref.get("row", 0)

    if step_idx >= len(step_results):
        raise ResolutionError(f"Reference to step {step_idx}, but only {len(step_results)} step(s) have run.")

    step_output = step_results[step_idx]

    if "rows" in step_output:  # SQL-style output
        rows = step_output["rows"]
        if row >= len(rows):
            raise ResolutionError(f"Step {step_idx} has no row {row} (only {len(rows)} row(s)).")
        if field not in rows[row]:
            raise ResolutionError(f"Step {step_idx} row {row} has no field {field!r}. Fields: {list(rows[row].keys())}")
        return float(rows[row][field])

    if "value" in step_output:  # calc-style output
        if field != "value":
            raise ResolutionError(f"Step {step_idx} is a calc step; only field 'value' is available, got {field!r}.")
        return float(step_output["value"])

    raise ResolutionError(f"Step {step_idx}'s output has no resolvable fields (got keys: {list(step_output.keys())}).")
